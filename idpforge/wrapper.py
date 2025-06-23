import pytorch_lightning as pl
import torch
import ml_collections as mlc
import os

from idpforge.model import IDPForge
from idpforge.loss import calc_loss, viol_loss
from idpforge.misc import output_to_pdb
from idpforge.utils.np_utils import coord_to_pdb
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils.validation_metrics import drmsd
from openfold.utils.tensor_utils import tensor_tree_map



class IDPForgeWrapper(pl.LightningModule):
    def __init__(self, config, denoiser):
        super(IDPForgeWrapper, self).__init__()
        self.config = config
        self.model = IDPForge(
                config["diffuse"]["n_tsteps"], 
                config["diffuse"]["n_tsteps_inf"],
                mlc.ConfigDict(config["model"]), 
            )
        self.loss_fn = calc_loss
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config["training"]["ema_decay"]
        )
        self.cached_weights = None
        self.denoiser = denoiser
        self.last_lr_step = -1
        self.save_pdbs = config["general"]["save_pdb"]
        self.epoch_save_freq = config["general"]["epoch_save_freq"]
        self.batch_save_freq = config["general"]["batch_save_freq"]
        self.save_hyperparameters(config)

    def _log(self, loss_breakdown, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in loss_breakdown.items():
            self.log(
                f"{phase}_{loss_name}", 
                indiv_loss, 
                on_step=train, on_epoch=(not train), logger=True,
                sync_dist=True
            )

            if train:
                self.log(
                    f"{phase}_{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True,
                    sync_dist=True
                )

    def training_step(self, batch, batch_idx):
        if self.ema.device != batch["ss"].device:
            self.ema.to(batch["ss"].device)

        in_batch = {k: batch[k] for k in ["sequence", "ss", "mask", "resi"]}
        if torch.rand(1).item() < 0.2:
            in_batch["ss"] = torch.zeros_like(batch["ss"]) + 7

        prev = None
        if self.config['model']['self_condition'] and torch.rand(1).item() < 0.5 and not torch.any(batch["T"] + 1 >= self.denoiser.diffuser.T):
            with torch.no_grad():
                prev = self.model((batch["T"] + 1).to(torch.long),  
                    batch["alpha_t+1"], batch["x_t+1"], in_batch, prev) 

        output = self.model(batch["T"].to(torch.long), 
                batch["alpha_t"], batch["x_t"], in_batch, prev)  
        output["sstype"] = batch["ss"]
        loss, loss_breakdown = self.loss_fn(output,
                batch["frame"], batch["coord"], batch["torsion"],
                self.config['training']['loss'])
                
        # Log it
        loss_breakdown["loss"] = loss
        self._log(loss_breakdown)
        if self.save_pdbs and (self.trainer.current_epoch + 1) % (self.epoch_save_freq * 2) == 0 and (batch_idx + 1) % (self.batch_save_freq * 10) == 0:
            self._log_pdbs(output, batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if self.cached_weights is None:
            clone_param = lambda t: t.detach().clone()
            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])
        
        in_batch = {k: batch[k] for k in ["sequence", "ss", "mask", "resi"]}
        output = self.model.recon(self.denoiser, batch["alpha_t"], batch["x_t"], in_batch) 
        output["sstype"] = batch["ss"]

        loss_breakdown = {}
        loss_breakdown["ca_drmsd"] = drmsd(output["positions"][-1][:, :, 1].detach(), 
            batch["coord"][:, :, 1], batch["mask"]).mean().item()
        v_l, v_mask = viol_loss(output, return_metric=True)
        loss_breakdown["violation"] = v_l.item()
        
        # Log it
        loss_breakdown["loss"] = loss_breakdown["ca_drmsd"] + 10*loss_breakdown["violation"]
        self._log(loss_breakdown, train=False)
        output["violation_per_residue"] = v_mask
        if self.save_pdbs and (self.trainer.current_epoch + 1) % self.epoch_save_freq == 0 and (batch_idx + 1) % self.batch_save_freq == 0:
            self._log_pdbs(output, batch, batch_idx, False) 


    def validation_epoch_end(self, _):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

    def configure_optimizers(self) -> torch.optim.Adam:
        # Ignored as long as a DeepSpeed optimizer is configured
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]

        print("total param size:", sum(p.numel() for p in trainable_parameters))
        optimizer = torch.optim.Adam(
            trainable_parameters, 
            lr=self.config["training"]["lr_scheduler"]["max_lr"], 
            eps=1e-5, 
        )
        
        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = self.config["training"]["lr_scheduler"]["max_lr"]

        lr_scheduler = AlphaFoldLRScheduler(optimizer, last_epoch=self.last_lr_step,
                                   **self.config["training"]["lr_scheduler"])
        if self.save_pdbs:
            self.pdb_dir = os.path.join(self.trainer.log_dir, "pred_pdbs")
            os.makedirs(self.pdb_dir, exist_ok=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "interval": "step",
                "frequency": 1
            }
        }
    
    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint["ema"]
        self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()
   
    @torch.no_grad()
    def _log_pdbs(self, output, batch, batch_idx, train=True):
        assert hasattr(self, "pdb_dir")
        out = {"positions": output["positions"][-1].detach().cpu().float()}
        for k in ["aatype", "sstype", "residue_index", "atom14_atom_exists",
                  "atom37_atom_exists", "residx_atom37_to_atom14", "violation_per_residue"]:
            if k in output:
                out[k] = output[k].detach().cpu().long()

        pdb = output_to_pdb(out, select_idx=[0])[0]
        pdbname = f"{self.trainer.current_epoch}_train{batch_idx + 1}_pred{batch['T'][0, 0]}.pdb" if train else f"{self.trainer.current_epoch}_val{batch_idx + 1}.pdb"
        pdb_out = os.path.join(self.pdb_dir, pdbname)
        with open(pdb_out, "w") as f:
            f.write(pdb)
                
        pdbname = f"{self.trainer.current_epoch}_train{batch_idx + 1}_true.pdb" if train else f"val{batch_idx + 1}_true.pdb"
        if train or not os.path.exists(os.path.join(self.pdb_dir, pdbname)):
            pdb = coord_to_pdb(batch["coord"].float().cpu().numpy()[0],
                            batch["sequence"].cpu().numpy()[0],
                            batch["mask"].cpu().numpy()[0].astype(bool),
                           )
            pdb_out = os.path.join(self.pdb_dir, pdbname)
            with open(pdb_out, "w") as f:
                f.write(pdb)
