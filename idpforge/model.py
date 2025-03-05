"""
Modified from ESM2
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import typing as T
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from esm.esmfold.trunk import FoldingTrunk, FoldingTrunkConfig
from esm.esmfold.misc import collate_dense_tensors

from openfold.data.data_transforms import make_atom14_masks, pseudo_beta_fn
from openfold.model.primitives import LayerNorm
from openfold.np.residue_constants import restype_num, restypes_with_x

from idpforge.misc import input_process, output_to_pdb
from idpforge.utils.definitions import sstype_num, sstypes
from idpforge.utils.tensor_utils import (
    get_chi_mask,
    xyz_to_t2d,
    sinusoidal_embedding,
    cdist
)

@dataclass
class IDPConfig:
    t_embed_dim = 32
    trunk = FoldingTrunkConfig()


class IDPForge(nn.Module):
    def __init__(self, n_tsteps, inf_tsteps, idp_config=None, **kwargs):
        super().__init__()
        
        self.cfg = idp_config if idp_config else IDPConfig(**kwargs)
        cfg = self.cfg
        c_s = cfg.trunk.sequence_state_dim
        c_z = cfg.trunk.pairwise_state_dim

        self.esm_s_mlp = nn.Sequential(
                LayerNorm(cfg.t_embed_dim + 8),
                nn.Linear(cfg.t_embed_dim + 8, c_s),
                nn.ReLU(),
                nn.Linear(c_s, c_s),
        )
        self.z_mlp = nn.Sequential(
                LayerNorm(cfg.t2d_params.DBINS*2 + 6 + 1),
                nn.Linear(cfg.t2d_params.DBINS*2 + 6 + 1, c_z),
                nn.ReLU(),
                nn.Linear(c_z, c_z),
            )
            
        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_tsteps, cfg.t_embed_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_tsteps, cfg.t_embed_dim)
        self.time_embed.requires_grad_(False)
        self.n_tsteps = n_tsteps
        self.inf_tsteps = inf_tsteps
        self.end_tsteps = kwargs.get("t_end", -1)

        # 0 is padding, N is mask (included).
        self.ss_embedding = nn.Embedding(sstype_num + 1, c_s, padding_idx=0)
        self.aa_embedding = nn.Embedding(restype_num + 2, c_s, padding_idx=0)
        self.trunk = FoldingTrunk(**cfg.trunk)

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return aa 

    def _ss_idx_to_esm_idx(self, ss, mask):
        ss = (ss + 1).masked_fill(mask != 1, 0)
        return ss

    def forward(
        self,
        t: torch.Tensor,
        alpha_t: torch.Tensor, 
        x_t: torch.Tensor,
        batch: T.Dict,
        prev_outputs: T.Optional[T.Dict] = None,
        num_recycles: T.Optional[int] = None,
        inplace_safe: bool = False,
        use_potential: bool =False
    ):
        """Runs a forward pass given input tokens. 

        Args:
            aa (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            ss (torch.Tensor): Tensor containing indices corresponding to secondary structure.
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
                ESMFold sometimes produces different samples when different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """
        device = self.device
        aa, ss, mask, residx = batch["sequence"], batch["ss"], batch["mask"], batch["resi"]
        
        if mask is None:
            mask = torch.ones_like(aa)
        if residx is None:
            residx = torch.arange(aa.shape[1], device=device).expand_as(aa)
        if inplace_safe:
            t, aa, ss, mask, residx = map(
                lambda x: x.to(device), (t, aa, ss, mask, residx))
                
        s_s_0 = torch.cat((self.time_embed(t), alpha_t.reshape(aa.shape + (8, ))), 
                dim=-1)
        s_s_0 = self.esm_s_mlp(s_s_0)
        s_s_0 = s_s_0 + self.aa_embedding(self._af2_idx_to_esm_idx(aa, mask)) 
        s_s_0 = s_s_0 + self.ss_embedding(self._ss_idx_to_esm_idx(ss, mask))

        B, L = aa.shape
        s_z_0 = s_s_0.new_zeros(B, L, L, self.cfg.trunk.pairwise_state_dim)
        s_z_0 += self.z_mlp(xyz_to_t2d(x_t, self.cfg.t2d_params, 
            aatype=aa, sampling=inplace_safe, pseudo_beta=use_potential)) 

        if prev_outputs is not None:
            s_s_0 = s_s_0 + self.trunk.recycle_s_norm(prev_outputs['s_s'])
            s_z_0 = s_z_0 + self.trunk.recycle_z_norm(prev_outputs['s_z'])
            s_z_0 = s_z_0 + self.trunk.recycle_disto(FoldingTrunk.distogram(
                prev_outputs["positions"][-1][:, :, :3],
                self.cfg.trunk.recycle_min_bin,
                self.cfg.trunk.recycle_max_bin,
                self.trunk.recycle_bins,
            ))
        
        structure: dict = self.trunk(
            s_s_0, s_z_0, aa, residx, mask, no_recycles=num_recycles
        )
        # Documenting what we expect:
        structure = {
            k: v for k, v in structure.items()
            if k in [
                "s_s", 
                "s_z",
                "frames",
                "sidechain_frames",
                "unnormalized_angles",
                "angles",
                "positions",
            ]
        }

        structure["aatype"] = aa       
        make_atom14_masks(structure)
        for k in ["atom14_atom_exists", "atom37_atom_exists"]:
            structure[k] *= mask.unsqueeze(-1)
        structure["residue_index"] = residx

        return structure

    @torch.no_grad()
    def recon(
        self,
        denoiser, 
        alpha_t: torch.Tensor,
        x_t: torch.Tensor,
        batch: T.Dict,
        potential = None,
        potential_grad_clip: T.Optional[float] = None,
        template_cfg: T.Optional[T.Dict] = None,
    ):
        aa, aa_mask = batch["sequence"], batch["mask"]
        d = x_t.dtype
        B, L = aa.shape[:2]
        torsion_mask = get_chi_mask(aa)
        prev_outputs = None
        
        x_t = x_t.to(self.device)
        alpha_t = alpha_t.to(self.device)
        for t in range(self.n_tsteps - 1, self.end_tsteps, -int(self.n_tsteps/self.inf_tsteps)):
            tstep = torch.full((B, L), t, dtype=torch.long, device=self.device)
            if template_cfg is not None:
                i, j = torch.where(template_cfg["mask"])
                tstep[i, j] = 0
                x_t[i, j] = template_cfg["coord"][i, j, :5, :]
                alpha_t[i, j] = template_cfg["torsion"][i, j]
            output = self.forward(
                tstep, alpha_t, x_t, batch, 
                prev_outputs=prev_outputs,
                inplace_safe=True,
                use_potential=potential is not None,
            )
            if t > 0:
                p_x0 = output["positions"][-1][:, :, :5].detach()
                p_tor0 = output["angles"][-1][:, :, 3:].detach()
                if potential is not None:
                    x_grad = potential.get_potential_gradients(p_x0)
                    p_x0 += torch.clamp(x_grad * self.potential_scaler(t + 1), max=potential_grad_clip)
                aa_diffusion_mask = aa_mask.cpu().numpy().astype(bool)
                x_t, alpha_t = denoiser.get_next_pose(
                            x_t.cpu().float().numpy(), 
                            p_x0.cpu().float().numpy(),     
                            alpha_t.cpu().float().numpy(), 
                            p_tor0.cpu().float().numpy(), 
                            t,
                            torsion_mask.cpu().numpy().astype(bool),
                            aa_diffusion_mask,
                            motiff_mask=template_cfg["mask"].cpu().numpy().astype(bool) if template_cfg else None)
                
                x_t = torch.tensor(x_t, dtype=d, device=self.device)
                alpha_t = torch.tensor(alpha_t, dtype=d, device=self.device)
                if self.cfg.self_condition:
                    prev_outputs = output
        return output


    @torch.no_grad()
    def sample(
        self,
        denoiser, 
        sequences: T.Union[str, T.List[str]],
        ss: T.Union[str, T.List[str]],
        alpha_t: T.Union[torch.Tensor, T.List[torch.Tensor]],
        x_t: T.Union[torch.Tensor, T.List[torch.Tensor]],
        residx: T.Optional[torch.Tensor] = None,
        potential_cfgs: T.Optional[T.Dict] = None,
        template_cfgs: T.Optional[T.Dict] = None,
        residue_index_offset: T.Optional[int] = 512,
        chain_linker: T.Optional[str] = "G" * 25,
        masking_pattern: T.Optional[torch.Tensor] = None,
    ):
        if isinstance(sequences, str):
            sequences = [sequences]
            ss = [ss]
            x_t = [x_t]
            alpha_t = [alpha_t]

        aatype, sstype, aa_mask, residx, linker_mask = input_process(
            sequences, ss, None, residx, residue_index_offset, chain_linker)
        if isinstance(x_t, T.List):
            x_t = collate_dense_tensors(x_t)
            alpha_t = collate_dense_tensors(alpha_t)
        if masking_pattern is not None:
            masking_pattern = collate_dense_tensors(masking_pattern)
        if potential_cfgs is not None:
            potential = self.initialize_potential(**potential_cfgs)

        batch = {"sequence": aatype, "ss": sstype, "mask": aa_mask, "resi": residx}
    
        output = self.recon(denoiser, alpha_t, x_t, batch, 
                potential=potential if potential_cfgs else None,
                potential_grad_clip=potential_cfgs.get("grad_clip", None) \
                        if potential_cfgs is not None else None,
                template_cfg=template_cfgs,
        )
        
        # deal with artificial linkers for multi-chains
        #output["backbone_exists"] *= linker_mask.to(self.device).unsqueeze(2)
        #output["chain_index"] = chain_index
        out = {"positions": output["positions"][-1].detach().cpu().float()}
        for k in ["aatype", "residue_index", "atom14_atom_exists", 
                  "atom37_atom_exists", "residx_atom37_to_atom14", "residx_atom14_to_atom37"]:
            out[k] = output[k].detach().cpu().long()
        out["sstype"] = sstype 
        return out
    
    def initialize_potential(self, potential_type, potential_cfg, timescale, time_schedule="linear", **kwargs):
        from idpforge.utils.potential import Potentials
        decay_types = {
            'constant': lambda t: timescale,
            'linear'  : lambda t: t / self.n_tsteps * timescale,
            'quadratic' : lambda t: t**2 / self.n_tsteps**2 * timescale
        }
        self.potential_scaler = decay_types[time_schedule]
        if len(potential_type) == 1:
            k = potential_type[0]
            return Potentials[k](**potential_cfg[k])
        return Potentials["multiple"](potential_cfg, kwargs["weights"])


    def set_chunk_size(self, chunk_size: T.Optional[int]):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        # Setting the value to None will return to default behavior, disable chunking.
        self.trunk.set_chunk_size(chunk_size)

    @property
    def device(self):
        return self.time_embed.weight.device
