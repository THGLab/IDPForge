# Auxilary file for preparing the LDR inputs

import numpy as np
import mdtraj as md
import argparse
import os
import sys

try:
    from idpforge.utils.np_utils import (
        process_pdb, assign_rama,
        get_chi_angles
    )
except ImportError:
    print("Error: Could not import from 'idpforge.utils.np_utils'.")
    print("       Ensure IDPForge is installed.")
    sys.exit(1)


_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Absolute seed-distance band (A)
SEED_FLOOR = 6.46
SEED_CEILING = 9.12


def normalize_variant_seq(variant_seq):
    """Validate/normalize an optional one-letter variant sequence (None if unset)."""
    if variant_seq is None:
        return None
    seq = variant_seq.strip().upper()
    if not seq:
        return None
    bad = sorted(set(seq) - _VALID_AA)
    if bad:
        raise ValueError(f"--variant_seq contains non-standard residues: {''.join(bad)}")
    return seq


def sample_fix_distance(batch_size, fixed_distance, fold_atoms=None, min_fold_dist=SEED_FLOOR):
    """Sample points on spheres of the given radii in random directions, dropping any within min_fold_dist of a folded CA."""
    if batch_size == 0:
        return np.empty((0, 3))

    theta = np.random.uniform(0, 2 * np.pi, batch_size)
    phi = np.random.uniform(0, np.pi, batch_size)
    x = fixed_distance * np.sin(phi) * np.cos(theta)
    y = fixed_distance * np.sin(phi) * np.sin(theta)
    z = fixed_distance * np.cos(phi)
    vectors = np.stack([x, y, z], axis=1)

    if fold_atoms is None or len(fold_atoms) == 0:
        return vectors

    dmin_sq = ((vectors[:, None, :] - fold_atoms[None, :, :]) ** 2).sum(-1).min(axis=1)
    return vectors[dmin_sq >= min_fold_dist ** 2]


def _truncate_to_size(template, mask, disorder_idx_list, max_residues,
                      fold_per_side=None):
    """Truncate the folded domain so the total system (IDR + kept fold) is <= max_residues."""
    coord = template["coord"]
    idr = ~mask
    n_idr = int(idr.sum())
    L = len(mask)

    idr_arr = np.asarray(disorder_idx_list)
    lo_idr, hi_idr = int(idr_arr.min()), int(idr_arr.max())
    has_above = (hi_idr + 1 < L) and bool(mask[hi_idr + 1:].any())
    has_below = (lo_idr > 0) and bool(mask[:lo_idr].any())
    n_sides = int(has_above) + int(has_below)

    # Per-junction folded-residue floor
    MIN_FOLD_PER_SIDE = 10
    budget = max(int(max_residues) - n_idr, 0)
    if n_sides == 0:
        per_side = 0
    elif fold_per_side is not None:
        per_side = int(fold_per_side)
    else:
        per_side = max(budget // n_sides, MIN_FOLD_PER_SIDE)
    _floor = int(fold_per_side) if fold_per_side is not None else MIN_FOLD_PER_SIDE
    idr_ceiling = int(max_residues) - _floor * max(n_sides, 1)
    if n_sides > 0 and n_idr > idr_ceiling:
        print(f"[mk_ldr] NOTE: IDR ({n_idr}) exceeds the single-step ceiling ({idr_ceiling}) for "
              f"{n_sides}-junction geometry at cap {max_residues}; keeping {MIN_FOLD_PER_SIDE} "
              f"fold/side (total {n_idr + MIN_FOLD_PER_SIDE * n_sides} > cap). Build the IDR in 2 "
              f"steps (split it; pin the first fragment as a pseudo-fold for the second).",
              flush=True)

    keep = idr.copy()
    if n_sides > 0:
        def walk(start, step):
            n, i = 0, start
            while 0 <= i < L and mask[i] and n < per_side:
                keep[i] = True
                n += 1
                i += step

        if has_above:
            walk(hi_idr + 1, 1)
        if has_below:
            walk(lo_idr - 1, -1)
    else:
        print(f"[mk_ldr] WARNING: IDR ({n_idr}) has no folded flank; nothing kept for stitching "
              f"(pure-IDP template).", flush=True)

    out = dict(template)
    kept_idx = np.where(keep)[0]
    out["trunc_orig_idx"] = kept_idx
    # Graft spec for Step-4 fold-completion
    fold_pos = np.where(mask)[0]
    if len(kept_idx) and len(fold_pos):
        out["graft_offset"] = int(kept_idx.min())
        out["graft_idr_ranges"] = np.array([[lo_idr + 1, hi_idr + 1]], dtype=int)
        out["graft_fold_range"] = np.array([int(fold_pos.min()) + 1, int(fold_pos.max()) + 1], dtype=int)
    if not keep.all():
        out["coord"] = coord[keep]
        out["torsion"] = template["torsion"][keep]
        out["mask"] = mask[keep]
        out["seq"] = "".join(np.array(list(template["seq"]))[keep])
        out["sec"] = "".join(np.array(list(template["sec"]))[keep])
        print(f"[mk_ldr] truncation: kept {int(keep.sum())}/{L} (IDR {n_idr} + fold "
              f"{int(keep.sum()) - n_idr}; cap {max_residues}).", flush=True)
    return out


def main(pdb, disorder_idx, nsample, variant_seq=None, seed_skew=0.3, min_fold_dist=SEED_FLOOR,
         seed_floor=SEED_FLOOR, seed_ceiling=SEED_CEILING, max_residues=None,
         append_seq=None, append_origin=None, prepend_seq=None, prepend_origin=None,
         fold_per_side=None):
    crd, seq = process_pdb(pdb)
    torsion = get_chi_angles(crd, seq)[0]
    torsion_vec = np.stack((np.sin(torsion), np.cos(torsion)), axis=-1)

    traj = md.load(pdb)
    dssp = md.compute_dssp(traj, simplified=True)[0]
    phis = md.compute_phi(traj)[1][0]
    psis = md.compute_psi(traj)[1][0]
    phis = np.concatenate(([-180], np.degrees(phis)))
    psis = np.concatenate((np.degrees(psis), [180]))
    rama = assign_rama(np.stack([phis, psis], axis=-1))
    encode = "".join([dssp[i] if dssp[i] in ["H", "E"] else rama[i] for i in range(len(dssp))])

    # Step-2 continuation: append a new C-terminal IDR tail
    if append_seq and prepend_seq:
        raise ValueError("append_seq and prepend_seq are mutually exclusive (one continuation "
                         "direction per Step-2 build).")
    if append_seq:
        nB = len(append_seq)
        L0 = len(seq)
        crd = np.concatenate([crd, np.zeros((nB,) + crd.shape[1:], dtype=crd.dtype)], axis=0)
        torsion_vec = np.concatenate(
            [torsion_vec, np.zeros((nB,) + torsion_vec.shape[1:], dtype=torsion_vec.dtype)], axis=0)
        seq = seq + append_seq
        encode = encode + "C" * nB
        disorder_idx = range(L0, L0 + nB)

    # Step-2 continuation, N-terminal direction: prepend a new IDR block
    if prepend_seq:
        nB = len(prepend_seq)
        crd = np.concatenate([np.zeros((nB,) + crd.shape[1:], dtype=crd.dtype), crd], axis=0)
        torsion_vec = np.concatenate(
            [np.zeros((nB,) + torsion_vec.shape[1:], dtype=torsion_vec.dtype), torsion_vec], axis=0)
        seq = prepend_seq + seq
        encode = "C" * nB + encode
        disorder_idx = range(0, nB)

    atom_mask = crd.sum(axis=-1) == 0

    disorder_idx_list = list(disorder_idx)
    if not disorder_idx_list:
         raise ValueError("Disorder index list is empty.")
         
    start_idx = disorder_idx_list[0]
    end_idx = disorder_idx_list[-1]

    # Define anchor point and exclusion zones
    if start_idx == 0: # N-terminal tail
        is_loop = False
        folded_center = crd[end_idx+1, 1]
        exclude_coords = [crd[end_idx+1:, 1]]
    elif end_idx == crd.shape[0] - 1: # C-terminal tail
        is_loop = False
        folded_center = crd[start_idx-1, 1]
        exclude_coords = [crd[:start_idx, 1]]
    else: # Loop
        is_loop = True
        folded_center = crd[[start_idx-1, end_idx+1], 1].mean(axis=0)
        exclude_coords = [crd[:start_idx, 1], crd[end_idx+1:, 1]]
    
    # Folded CA keep-out surface
    fold_atoms = np.concatenate([c for c in exclude_coords if c.shape[0] > 0], axis=0).copy()
    fold_atoms = fold_atoms[np.any(fold_atoms != 0, axis=1)]

    # Detach the anchor before the in-place shift
    folded_center = np.array(folded_center)
    crd -= folded_center
    shifted_fold_atoms = fold_atoms - folded_center
    
    # Seed radii skewed on [floor, ceiling]
    def _skew_radii(n):
        u = np.random.uniform(0, 1, n)
        if seed_skew <= 0:
            b = np.zeros(n)
        elif seed_skew >= 1:
            b = np.ones(n)
        else:
            b = u ** ((1.0 - seed_skew) / seed_skew)
        return seed_floor + b * (seed_ceiling - seed_floor)

    # Oversample, dropping centers within min_fold_dist of the fold, until nsample remain
    found, n_have = [], 0
    for _ in range(40):
        if n_have >= nsample:
            break
        n = max(nsample - n_have, 1) * 20
        pts = sample_fix_distance(n, _skew_radii(n), shifted_fold_atoms, min_fold_dist)
        if len(pts):
            found.append(pts)
            n_have += len(pts)
    noise_init = np.concatenate(found, axis=0) if found else np.array([[seed_ceiling, 0.0, 0.0]])
    if len(noise_init) < nsample:
        noise_init = np.tile(noise_init, (int(np.ceil(nsample / len(noise_init))), 1))

    # Zero-out missing atoms
    i, j = np.where(atom_mask)
    crd[i, j] = 0
    
    # Create template
    template = {"coord": crd, "torsion": torsion_vec, "sec": encode, "seq": seq}

    mask = np.ones(len(crd), dtype=bool)
    mask[disorder_idx_list] = False
    template["mask"] = mask
    template["coord_offset"] = noise_init[:nsample]

    # Optional variant-sequence override
    if variant_seq is not None:
        if len(variant_seq) != len(disorder_idx_list):
            raise ValueError(
                f"--variant_seq length ({len(variant_seq)}) must equal the disordered "
                f"region length ({len(disorder_idx_list)} residues, "
                f"{start_idx + 1}-{end_idx + 1})."
            )
        seq_chars = list(template["seq"])
        for offset, pos in enumerate(disorder_idx_list):
            seq_chars[pos] = variant_seq[offset]
        template["seq"] = "".join(seq_chars)

    # Do not truncate a Step-2 continuation
    if max_residues is not None and not append_seq and not prepend_seq:
        template = _truncate_to_size(template, mask, disorder_idx_list,
                                     max_residues, fold_per_side=fold_per_side)
        # Stamp the source fold PDB for Step-4 stitching
        if isinstance(template, dict) and "graft_offset" in template:
            template["graft_static_pdb"] = os.path.basename(str(pdb))

    # Step-2 origin tag
    if append_seq and append_origin is not None:
        template["step2_a_conformer"] = str(append_origin.get("a_conformer", ""))
        template["step2_a_len"] = int(len(seq) - len(append_seq))
        template["step2_b_len"] = int(len(append_seq))
        template["step2_direction"] = "append"
    if prepend_seq and prepend_origin is not None:
        template["step2_a_conformer"] = str(prepend_origin.get("a_conformer", ""))
        template["step2_a_len"] = int(len(seq) - len(prepend_seq))
        template["step2_b_len"] = int(len(prepend_seq))
        template["step2_direction"] = "prepend"

    return template


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare LDR inputs from a PDB file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', help="Input PDB file.")
    parser.add_argument('disorder_domain', help="Specify residue number for disordered region (1-index, both ends inclusive). Example: 1-15 or 38-129")
    parser.add_argument('output', help="Output .npz file.")
    parser.add_argument('--nconf', default=400, type=int, help="Number of seed points to sample.")
    parser.add_argument('--seed_skew', default=0.3, type=float,
        help="Skew of the seed distance across the absolute band [seed_floor, seed_ceiling]: "
             "0 = all at the floor (6.46 A, the empirically best seed; reproduces abs646), "
             "1 = all at the ceiling, 0.5 = uniform. Default 0.3 concentrates near the floor "
             "while keeping spread for ensemble breadth.")
    parser.add_argument('--seed_floor', default=SEED_FLOOR, type=float,
        help=f"Absolute seed-band floor in A (length-INDEPENDENT). Default {SEED_FLOOR} = "
             f"1 CA-CA + 2 C-N ~= IDR cloud mean radius (cloud tangent to fold).")
    parser.add_argument('--seed_ceiling', default=SEED_CEILING, type=float,
        help=f"Absolute seed-band ceiling in A. Default {SEED_CEILING} = 1 CA-CA + 4 C-N ~= cloud "
             f"mean + 1 radial std (cloud just clears fold). Raise toward 10.45 for more breadth.")
    parser.add_argument('--min_fold_dist', default=SEED_FLOOR, type=float,
        help=f"Keep-out: reject cloud-center placements within this distance of the nearest folded CA. "
             f"This is a CLOUD-CENTER standoff (~cloud radius), NOT an atom-atom clash. Default {SEED_FLOOR}.")
    parser.add_argument('--variant_seq', default=None, type=str,
        help="Optional one-letter amino-acid sequence to overwrite the identities of the "
             "disordered region (e.g. a chimera variant tail). Must match the length of "
             "disorder_domain. The IDR is diffused, so only residue identity is used.")
    parser.add_argument('--max_residues', default=None, type=int,
        help="Hard cap on TOTAL system size (IDR + fold). IDPForge is in-distribution only "
             "for systems <= ~200, so set 200 for the large folds. Keeps the IDR + the "
             "contiguous junction-adjacent fold block sized to the budget (cap - IDR len); "
             "drops the rest. A floor of 50 folded residues is always kept (even if that "
             "exceeds the cap) so Step-4 stitching stays viable. Saves trunc_orig_idx + graft "
             "spec for stitch-back onto the full fold.")
    parser.add_argument('--fold_per_side', default=None, type=int,
        help="Keep EXACTLY this many folded residues per junction (the 'IDR + N fold' mode), "
             "ignoring the --max_residues budget: smaller and more in-distribution than filling "
             "the budget (e.g. 50 -> IDR + 50 fold). Default (unset) fills the budget, floored at 50.")
    parser.add_argument('--append_seq', default=None, type=str,
        help="Step-2 continuation: append this one-letter sequence as a NEW C-terminal IDR tail "
             "(fragment B) grown off the input structure (fragment A, treated as the pinned "
             "pseudo-fold). Overrides disorder_domain (the IDR becomes the appended range). Grows "
             "an IDR beyond the single-step ceiling in 2 steps.")
    parser.add_argument('--append_origin', default=None, type=str,
        help="Step-2 only: identifier (e.g. path) of the Step-A conformer fragment A came from, "
             "stamped into the template so Step 4 can chain fold <- A <- B.")
    parser.add_argument('--prepend_seq', default=None, type=str,
        help="Step-2 continuation, N-terminal direction: prepend this one-letter sequence as a NEW "
             "N-terminal IDR block (fragment B) grown off the input structure (fragment A, the pinned "
             "pseudo-fold). The N-terminal counterpart of --append_seq. Overrides disorder_domain.")
    parser.add_argument('--prepend_origin', default=None, type=str,
        help="Step-2 only: identifier of the Step-A conformer fragment A came from (prepend direction).")
    args = parser.parse_args()

    try:
        i, j = args.disorder_domain.split("-")
        disorder_range = range(int(i)-1, int(j))
        variant_seq = normalize_variant_seq(args.variant_seq)
        append_seq = normalize_variant_seq(args.append_seq)
        append_origin = {"a_conformer": args.append_origin or args.input} if append_seq else None
        prepend_seq = normalize_variant_seq(args.prepend_seq)
        prepend_origin = {"a_conformer": args.prepend_origin or args.input} if prepend_seq else None
        out = main(args.input, disorder_range, args.nconf, variant_seq=variant_seq,
                   seed_skew=args.seed_skew, min_fold_dist=args.min_fold_dist,
                   seed_floor=args.seed_floor, seed_ceiling=args.seed_ceiling,
                   max_residues=args.max_residues,
                   append_seq=append_seq, append_origin=append_origin,
                   prepend_seq=prepend_seq, prepend_origin=prepend_origin,
                   fold_per_side=args.fold_per_side)
        np.savez(args.output, **out)
        print(f"Successfully generated {args.output} with {args.nconf} seed points "
              f"(absolute band [{args.seed_floor}, {args.seed_ceiling}] A, skew={args.seed_skew}, "
              f"keep-out={args.min_fold_dist} A).")
    except Exception as e:
        print(f"Error processing {args.input} with {args.disorder_domain}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)