# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as T
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn as nn

import esm
from esm import Alphabet
from openfold.np.residue_constants import restype_num, restypes_with_x



load_fn = esm.pretrained.load_model_and_alphabet
esm_registry = {
    "esm2_8M": partial(load_fn, "esm2_t6_8M_UR50D_500K"),
    "esm2_8M_270K": esm.pretrained.esm2_t6_8M_UR50D,
    "esm2_35M": partial(load_fn, "esm2_t12_35M_UR50D_500K"),
    "esm2_35M_270K": esm.pretrained.esm2_t12_35M_UR50D,
    "esm2_150M": partial(load_fn, "esm2_t30_150M_UR50D_500K"),
    "esm2_150M_270K": partial(load_fn, "esm2_t30_150M_UR50D_270K"),
    "esm2_650M": esm.pretrained.esm2_t33_650M_UR50D,
    "esm2_650M_270K": partial(load_fn, "esm2_t33_650M_270K_UR50D"),
    "esm2_3B": esm.pretrained.esm2_t36_3B_UR50D,
    "esm2_3B_270K": partial(load_fn, "esm2_t36_3B_UR50D_500K"),
    "esm2_15B": esm.pretrained.esm2_t48_15B_UR50D,
}

class ESM_preprocess(nn.Module):
    def __init__(self, esm_type, use_esm_attn_map):
        super().__init__()
        self.esm, self.esm_dict = esm_registry.get(esm_type)()
        self.esm.half()
        self.esm.requires_grad_(False) 
        self.use_esm_attn_map = use_esm_attn_map
        self.register_buffer("af2_to_esm", self._af2_to_esm(self.esm_dict))
        self.af2_to_esm.requires_grad_(False)
    
    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [
            d.get_idx(v) for v in restypes_with_x]
        return torch.tensor(esm_reorder)
    
    def _compute_language_model_representations(
        self, esmaa: torch.Tensor
    ) -> torch.Tensor:
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = esmaa.new_full((batch_size, 1), bosi)
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=self.use_esm_attn_map,
        )
        esm_s = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        esm_s = esm_s[:, 1: -1]  # B, L, nLayers, C
        esm_z = (
            res["attentions"].permute(0, 4, 3, 1, 2).flatten(3, 4)[:, 1:-1, 1:-1, :]
            if self.use_esm_attn_map
            else None
        )
        return esm_s, esm_z

    def _mask_inputs_to_esm(self, esmaa, pattern):
        new_esmaa = esmaa.clone()
        new_esmaa[pattern == 0] = self.esm_dict.mask_idx
        return new_esmaa

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return aa
    
    def forward(self, aa, mask, masking_pattern=None):
        device = aa.device
        aa_embed = self._af2_idx_to_esm_idx(aa, mask)
        esmaa = self.af2_to_esm[aa_embed.cpu()]
        if masking_pattern is not None:
            esmaa = self._mask_inputs_to_esm(esmaa, masking_pattern.cpu())
        esm_s, esm_z = self._compute_language_model_representations(esmaa)
        return esm_s.detach().to(device), \
                None if esm_z is None else esm_z.detach().to(device)


