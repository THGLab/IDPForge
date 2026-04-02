import numpy as np


def calc_rg_with_mask(ca_coords, mask):
    """
    Compute radius of gyration using CA atoms and mask.

    Args:
        ca_coords: ndarray of shape [batch, nres, 3]
        mask: ndarray of shape [batch, nres], 1 for valid residues

    Returns:
        ndarray of shape [batch] containing radius of gyration per batch
    """
    mask = mask.astype(np.float32)
    mask_expanded = mask[:, :, None]

    mask_sum = np.sum(mask, axis=1, keepdims=True)
    mask_sum_clipped = np.clip(mask_sum, a_min=1.0, a_max=None)

    center = np.sum(ca_coords * mask_expanded, axis=1) / mask_sum_clipped

    diff = ca_coords - center[:, None, :]
    sq_dist = np.sum(diff**2, axis=-1)

    rg_sq = np.sum(sq_dist * mask, axis=1) / np.clip(np.sum(mask, axis=1), 1.0, None)
    rg = np.sqrt(rg_sq)

    return rg


def rg_dist_per_group(pred_crds, true_rgs, sequence, mask):
    """
    Compute divergence between predicted and true Rg distributions grouped by sequence.

    Groups conformers by sequence identity (hash), computes mean Rg per group,
    and returns the mean absolute difference across groups.
    """
    seq_hashes = [hash(seq.tobytes()) for seq in sequence]

    pred_rg = calc_rg_with_mask(pred_crds, mask)
    results = []
    unique_hashes = np.unique(seq_hashes)
    for h in unique_hashes:
        idx = np.array([sh == h for sh in seq_hashes])
        if np.sum(idx) == 0:
            continue
        pred_group = pred_rg[idx]
        true_group = true_rgs[idx]

        diff = np.abs(np.mean(pred_group) - np.mean(true_group))
        results.append(diff)
    return np.mean(results)
