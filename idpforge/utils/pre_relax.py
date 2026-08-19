"""Pre-relaxation screening for diffusion outputs (generation stage)."""

import numpy as np


_JUNCTION_CA_THRESHOLD = 6.46
_BACKBONE_CA_THRESHOLD = 9.12


def check_backbone_continuity(atom_positions, viol_mask, residue_index=None,
                               junction_threshold=_JUNCTION_CA_THRESHOLD,
                               backbone_threshold=_BACKBONE_CA_THRESHOLD):
    """Check every consecutive CA-CA peptide bond along the chain, before relaxation."""
    CA_IDX = 1
    details = []
    for i in range(len(viol_mask) - 1):
        if residue_index is not None and int(residue_index[i + 1]) - int(residue_index[i]) != 1:
            continue
        ca_i = atom_positions[i, CA_IDX]
        ca_j = atom_positions[i + 1, CA_IDX]
        dist = float(np.linalg.norm(ca_i - ca_j))
        is_junction = bool(viol_mask[i] != viol_mask[i + 1])
        thr = junction_threshold if is_junction else backbone_threshold
        details.append({
            "res_i": i, "res_j": i + 1, "distance": dist,
            "kind": "junction" if is_junction else "backbone",
            "pass": dist <= thr,
        })
    return all(d["pass"] for d in details), details
