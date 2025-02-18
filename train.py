"""
Script to run training of IDPForge
created by OZ, 11/12/24
"""

import os
import numpy as np
import argparse
import torch
import yaml

from idpforge.loader import IDPloader
from idpforge.wrapper import IDPForgeWrapper
from idpforge.utils.diff_utils import Diffuser, Denoiser
from openfold.utils.callbacks import EarlyStoppingVerbose

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium')

def main(args):
    seed_everything(args.seed)
    settings = yaml.safe_load(open(args.model_config_path, "r"))
    # data
    diffuser = Diffuser(settings["diffuse"]["n_tsteps"])
    denoiser = Denoiser(settings["diffuse"]["n_tsteps_inf"], diffuser)
    data_module = IDPloader([settings["data"]["path"] + f"{n}.pkl" for n in settings["data"]["idpseq"]], 
                    diffuser=diffuser,
                    train_size=settings["data"]["train_size"], 
                    val_size=settings["data"]["val_size"],
                    tr_batch_size=settings['data']['tr_batch_size'],
                    val_batch_size=settings['data']['val_batch_size'],
                    test_batch_size=settings['data']['test_batch_size'],
                    )
    data_module.setup("fit")
    
    # model
    model = IDPForgeWrapper(settings, denoiser=denoiser)
    callbacks = []

    ckpt_path = None
    if args.resume_from_ckpt and not args.load_weights_only:
        if (os.path.isdir(args.resume_from_ckpt)):  
            last_global_step = get_global_step_from_checkpoint(args.resume_from_ckpt)
        else:
            sd = torch.load(args.resume_from_ckpt, map_location="cpu")
            last_global_step = int(sd['global_step'])
        model.resume_last_lr_step(last_global_step)
    
    if args.load_weights_only:
        sd = torch.load(args.resume_from_ckpt, map_location="cpu")
        model.load_state_dict(sd["state_dict"])
        #model.ema.load_state_dict(sd["ema"])
        print("Loading model weights from", args.resume_from_ckpt)
    else:
        ckpt_path = args.resume_from_ckpt

    mc = ModelCheckpoint(
        monitor="val_ca_drmsd",
        every_n_epochs=1,
        auto_insert_metric_name=False,
        save_top_k=3,
        filename='{epoch}-{step}',
        save_last=True,
    )
    callbacks.append(mc)
    
    if args.early_stopping:
        es = EarlyStoppingVerbose(
            monitor="val_loss",
            min_delta=settings["early_stopping"]["min_delta"],
            patience=settings["early_stopping"]["patience"],
            verbose=False,
            mode="min",
            check_finite=True,
            strict=True,
        )
        callbacks.append(es)
    
    if args.log_lr:
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
    
    logger = TensorBoardLogger(settings["general"]["output"], 
        name="lightning_logs" if "run_name" not in settings["general"] else settings["general"]["run_name"],
        version=args.run_version,
        )
    trainer = Trainer(
        strategy="ddp",
        callbacks=callbacks,
        logger=logger,
        **settings["training"]["trainer"],
    )
    trainer.fit(
        model, 
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config_path", type=str, default="config.yml",
        help="Path to model & trainer config.")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--run_version", type=int, default=None,
    )
    parser.add_argument(
        "--early_stopping", action="store_true", default=False,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--load_weights_only", action="store_true", default=False,
        help="Whether load model weights only but not optimizer status"
    )
    parser.add_argument(
        "--log_lr", action="store_true", default=True,
        help="Whether to log the actual learning rate"
    )

    args = parser.parse_args()

    main(args)
