# Auxilary file for preparing the LDR inputs with flexible domains

import numpy as np
import mdtraj as md
import argparse
import sys
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from idpforge.utils.np_utils import process_pdb, assign_rama, get_chi_angles

_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

_MOMENT = np.sqrt(10.0 / 9.0)


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


def random_rotation_matrix():
    """Generate a random 3D rotation matrix (uniformly sampled from SO(3))."""
    return Rotation.random().as_matrix()

def random_rotate_translate_coords_scaled(
    coords_centered, ref_center, d
):
    """Apply a random rotation and random-direction translation, then move to 'ref_center'."""
    R = random_rotation_matrix()
    
    rotated = np.einsum('nij,jk->nik', coords_centered, R.T)
    
    direction = np.random.randn(3)
    if np.linalg.norm(direction) > 1e-6:
        direction /= np.linalg.norm(direction)
    translation = direction * d
    
    return rotated + translation + ref_center

def random_nonclashing_transform_scaled(
    fold2_centered,
    fold1_coords_rotated, 
    d_sample,
    max_attempts=500, 
    min_dist=3.8
):
    """Find a non-clashing placement for fold2 relative to fold1 within max_attempts tries."""

    flat_fold1 = fold1_coords_rotated.reshape(-1, 3)
    flat_fold1 = flat_fold1[np.any(flat_fold1 != 0, axis=1)]
    if flat_fold1.shape[0] == 0:
        ref_center = fold1_coords_rotated[-1, 1]
        return True, random_rotate_translate_coords_scaled(fold2_centered, ref_center, d_sample)

    fold1_tree = KDTree(flat_fold1)
    ref_center = fold1_coords_rotated[-1, 1]

    for i in range(max_attempts):
        if (i + 1) % 100 == 0: 
            print(f"       ... Sampling linker pose (Try {i + 1}/{max_attempts})", flush=True)

        new_coords_fold2 = random_rotate_translate_coords_scaled(
            fold2_centered, ref_center, d_sample
        )

        flat_fold2 = new_coords_fold2.reshape(-1, 3)
        flat_fold2 = flat_fold2[np.any(flat_fold2 != 0, axis=1)]
        
        if flat_fold2.shape[0] == 0:
            return True, new_coords_fold2
        dists, _ = fold1_tree.query(flat_fold2, k=1)
        
        if np.min(dists) > min_dist:
            return True, new_coords_fold2
            
    return False, None


def sample_fold_orientation(
    fold1_centered,
    center_fold1,
    fold2_centered,
    idr_len,
    min_dist=3.8,
    max_attempts=100,
    max_outer_loops=50,
    ree_scale=1.0
):
    """Find a non-clashing orientation for two domains."""
    # Flory-based target end-to-end distance
    d = 5.51 * idr_len ** 0.588 * ree_scale
    
    # Absolute physical maximum
    max_physical_limit = idr_len * 3.8

    for i in range(max_outer_loops):
        if (i + 1) % 10 == 0:
            print(f"       ... Sampling domain orientation (Outer loop {i + 1}/{max_outer_loops})", flush=True)
    
        R = random_rotation_matrix()
        
        rotated_fold1 = np.einsum('nij,jk->nik', fold1_centered, R.T)
        new_fold1 = rotated_fold1 + center_fold1
        
        # Sample distance, clamped to the physical limit
        mu = d / _MOMENT
        raw_dist = np.random.normal(mu, mu / 3)
        dist_sample = min(abs(raw_dist), max_physical_limit)

        success, new_fold2 = random_nonclashing_transform_scaled(
            fold2_centered, new_fold1, dist_sample, 
            max_attempts=max_attempts, 
            min_dist=min_dist
        )
            
        if success:
            return new_fold1, new_fold2 

    print(f"       Warning: sample_fold_orientation failed to find a non-clashing pose after {max_outer_loops} outer loops. Returning a *clashing* pose as fallback.", flush=True)
    
    _mu = d / _MOMENT
    raw_fallback_d = np.random.normal(_mu, _mu / 3)
    fallback_d = min(abs(raw_fallback_d), max_physical_limit)
    
    R = random_rotation_matrix()
    rotated_fold1 = np.einsum('nij,jk->nik', fold1_centered, R.T)
    new_fold1 = rotated_fold1 + center_fold1
    ref_center = new_fold1[:, 1].mean(axis=0)

    fallback_fold2 = random_rotate_translate_coords_scaled(fold2_centered, ref_center, fallback_d)
    return new_fold1, fallback_fold2


def _truncate_linker_domains(crd, seq, encode, torsion_vec, linker_start, linker_end,
                             max_residues, min_fold_per_side=10, fold_per_side=None):
    """Size-cap a linker template: keep the linker + a junction-adjacent block of each flanking domain."""
    L = len(seq)
    n_linker = linker_end - linker_start + 1
    n_f1, n_f2 = linker_start, L - (linker_end + 1)
    budget = max(int(max_residues) - n_linker, 0)
    if fold_per_side is not None:
        per_side = int(fold_per_side)
    else:
        per_side = max(budget // 2, min_fold_per_side)
    keep_f1, keep_f2 = min(per_side, n_f1), min(per_side, n_f2)
    if keep_f1 == n_f1 and keep_f2 == n_f2:
        return crd, seq, encode, torsion_vec, linker_start, linker_end, None

    idr_ceiling = int(max_residues) - min_fold_per_side * 2
    if n_linker > idr_ceiling:
        print(f"[mk_flex] NOTE: linker ({n_linker}) exceeds the single-step ceiling ({idr_ceiling}) "
              f"at cap {max_residues}; keeping {min_fold_per_side}/junction (total "
              f"{n_linker + 2 * min_fold_per_side} > cap). Build the linker in 2 steps.", flush=True)

    f1_lo = linker_start - keep_f1
    f2_hi = linker_end + keep_f2
    keep = np.arange(f1_lo, f2_hi + 1)
    graft = {
        "offset": int(f1_lo),
        "idr_ranges": [[linker_start + 1, linker_end + 1]],
        "fold_ranges": [[1, linker_start], [linker_end + 2, L]],
    }
    print(f"[mk_flex] truncation: kept {len(keep)}/{L} (fold1 {keep_f1} + linker {n_linker} + "
          f"fold2 {keep_f2}; cap {max_residues}).", flush=True)
    return (crd[keep], "".join(seq[i] for i in keep), "".join(encode[i] for i in keep),
            torsion_vec[keep], linker_start - f1_lo, linker_end - f1_lo, graft)


def main(input_pdb, disorder_idx, nsample, variant_seq=None, max_residues=None,
         ree_scale=1.0, min_fold_per_side=10, fold_per_side=None, **kwargs):
    """Generate the multi-pose template .npz file."""
    # Load PDB and compute features
    crd, seq = process_pdb(input_pdb)
    torsion = get_chi_angles(crd, seq)[0]
    torsion_vec = np.stack((np.sin(torsion), np.cos(torsion)), axis=-1)
    
    print("       Loading PDB and calculating features (mdtraj)...", flush=True)
    traj = md.load(input_pdb)
    
    num_residues = traj.topology.n_residues
    disorder_range_list_check = list(disorder_idx) 
    
    if disorder_range_list_check[0] == 0 or disorder_range_list_check[-1] == (num_residues - 1):
        raise ValueError(f"Error: {disorder_idx} appears to be an N/C-terminal tail, not a linker. This script is only for linkers.")
    
    dssp = md.compute_dssp(traj, simplified=True)[0]
    phis = md.compute_phi(traj)[1][0]
    psis = md.compute_psi(traj)[1][0]
    phis = np.concatenate(([-180], np.degrees(phis)))
    psis = np.concatenate((np.degrees(psis), [180]))
    
    rama = assign_rama(np.stack([phis, psis], axis=-1))
    encode = "".join([dssp[i] if dssp[i] in ["H", "E"] else rama[i] for i in range(len(dssp))])

    # Identify domains and linker
    print("       Identifying domains and linker region...", flush=True)
    disorder_range = list(disorder_idx) 
    
    if disorder_range[0] > 0 and (encode[disorder_range[0]-1] in ["H", "E"] or "EE" in encode[max(0, disorder_range[0]-4): disorder_range[0]]):
        disorder_range = disorder_range[2:]
    
    if not disorder_range: raise ValueError("Disorder index trimming (start) resulted in an empty linker region.")

    if disorder_range[-1] < len(encode) - 1 and (encode[disorder_range[-1]+1] in ["H", "E"] or "EE" in encode[disorder_range[-1]: min(len(encode), disorder_range[-1]+4)]):
        disorder_range = disorder_range[:-2]
    
    if not disorder_range: raise ValueError("Disorder index trimming (end) resulted in an empty linker region.")
          
    linker_start_idx = disorder_range[0]
    linker_end_idx = disorder_range[-1]

    # Size-cap truncation
    graft_spec = None
    if max_residues is not None:
        crd, seq, encode, torsion_vec, linker_start_idx, linker_end_idx, graft_spec = \
            _truncate_linker_domains(crd, seq, encode, torsion_vec, linker_start_idx,
                                     linker_end_idx, max_residues, min_fold_per_side,
                                     fold_per_side=fold_per_side)
        disorder_range = list(range(linker_start_idx, linker_end_idx + 1))
          
    fold1_coords = crd[:linker_start_idx]
    fold2_coords = crd[linker_end_idx + 1:]
    
    # Centre each domain on its junction CA
    center_fold1 = fold1_coords[-1, 1]
    fold1_centered = fold1_coords - center_fold1

    center_fold2 = fold2_coords[0, 1]
    fold2_centered = fold2_coords - center_fold2
    
    # Generate domain orientations
    new_coords = np.tile(crd, (nsample, 1, 1, 1))
    atom_mask = new_coords.sum(axis=-1) == 0
    
    print(f"       Generating {nsample} random domain orientations (min_dist=3.8)...", flush=True)
    for i in range(nsample):
        new_fold1, new_fold2 = sample_fold_orientation(
            fold1_centered,
            center_fold1,
            fold2_centered,
            len(disorder_range),
            ree_scale=ree_scale
        )
        
        new_coords[i, :linker_start_idx] = new_fold1
        new_coords[i, linker_end_idx + 1:] = new_fold2
        new_coords[i, linker_start_idx: linker_end_idx + 1] = 0

    # Centre on the junction-residue midpoint
    center = new_coords[:, [linker_start_idx - 1, linker_end_idx + 1], 1].mean(axis=1) + np.random.uniform(0, len(disorder_range) / 5, size=(nsample, 3))
    new_coords -= center[:, None, None, :]
    
    i_ns, i_res, i_atom = np.where(atom_mask)
    new_coords[i_ns, i_res, i_atom] = 0

    # Build the template
    template = {"coord": new_coords, "torsion": torsion_vec, "sec": encode, "seq": seq}

    mask = np.ones(len(seq), dtype=bool)
    mask[disorder_range] = False
    template["mask"] = mask

    # Record the graft spec
    if graft_spec is not None:
        template["graft_offset"] = int(graft_spec["offset"])
        template["graft_idr_ranges"] = np.array(graft_spec["idr_ranges"], dtype=int)
        template["graft_fold_ranges"] = np.array(graft_spec["fold_ranges"], dtype=int)

    # Optional variant-sequence override
    if variant_seq is not None:
        if len(variant_seq) != len(disorder_range_list_check):
            raise ValueError(
                f"--variant_seq length ({len(variant_seq)}) must equal the disorder_domain "
                f"length ({len(disorder_range_list_check)} residues, "
                f"{disorder_range_list_check[0] + 1}-{disorder_range_list_check[-1] + 1})."
            )
        trunc_off = graft_spec["offset"] if graft_spec is not None else 0
        seq_chars = list(template["seq"])
        for k, pos in enumerate(disorder_range_list_check):
            new_pos = pos - trunc_off
            if 0 <= new_pos < len(seq_chars):
                seq_chars[new_pos] = variant_seq[k]
        template["seq"] = "".join(seq_chars)

    return template

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a multi-pose .npz template for flexible linker modeling.")
    parser.add_argument('input', help="Input PDB file path.")
    parser.add_argument('disorder_domain', help="Specify residue number for disordered region (1-index, both ends inclusive). Example: 38-129")
    parser.add_argument('output', help="Output .npz file path.")
    parser.add_argument('--nconf', default=200, type=int, help="Number of random domain orientations to sample.")
    parser.add_argument('--variant_seq', default=None, type=str,
        help="Optional one-letter amino-acid sequence to overwrite the identities of the "
             "disordered linker (e.g. a chimera variant). Must match the length of "
             "disorder_domain. The linker is diffused, so only residue identity is used.")
    parser.add_argument('--max_residues', default=None, type=int,
        help="Hard cap on TOTAL system size (linker + both fold flanks) to stay in IDPForge's "
             "in-distribution range (<= ~200). Keeps the linker + the junction-adjacent block of "
             "EACH domain, sized to (cap - linker)//2 per side; drops the distal domain parts. A "
             "floor of --min_fold_per_side folded residues PER JUNCTION is always kept (single-step "
             "linker ceiling = cap - 2*min_fold_per_side). Records the two-domain graft spec for "
             "Step-4 stitch-back.")
    parser.add_argument('--ree_scale', default=1.0, type=float,
        help="Scale on the Flory end-to-end target Re = 5.51*N^0.588 used to separate the two "
             "domains (1.0 = physical random-coil separation). Lower it (<1) to tighten the "
             "inter-domain distance if far-anchor bridging/yield is poor (linker analog of the "
             "tail d/2 reach fix).")
    parser.add_argument('--fold_per_side', default=None, type=int,
        help="Keep EXACTLY this many folded residues per junction, ignoring the --max_residues "
             "budget ('linker + N fold' mode). Same meaning and same flag name as "
             "mk_ldr_template's --fold_per_side, so loops and linkers stay comparable; Step 2 "
             "passes config.TEMPLATE_FOLD_PER_SIDE to both. Without it the linker fills the "
             "budget, which for a 146-residue linker under a 200 cap means 27 fold residues per "
             "side against a loop's 10.")
    parser.add_argument('--min_fold_per_side', default=10, type=int,
        help="Per-junction folded-residue floor kept during truncation (default 10). Matches "
             "mk_ldr's --fold_per_side / config.TEMPLATE_FOLD_PER_SIDE: 50 was found to deviate "
             "too far from the original tail behaviour, and a big folded block also drives IDR "
             "over-expansion. Keeping 10 puts the linker ceiling at max_residues - 20 = 180 aa, "
             "the same as a loop, instead of 100.")

    args = parser.parse_args()

    try:
        i, j = args.disorder_domain.split("-")
        disorder_range = range(int(i)-1, int(j))
        variant_seq = normalize_variant_seq(args.variant_seq)

        out = main(args.input, disorder_range, args.nconf, variant_seq=variant_seq,
                   max_residues=args.max_residues, ree_scale=args.ree_scale,
                   min_fold_per_side=args.min_fold_per_side,
                   fold_per_side=args.fold_per_side)
        
        np.savez(args.output, **out)
        print(f"Successfully created flexible template: {args.output}", flush=True)
        
    except FileNotFoundError:
        print(f"Error: Input PDB file not found at {args.input}", file=sys.stderr)
        sys.exit(1)
    except ImportError:
         print(f"Error: Missing imports from 'idpforge.utils.np_utils'.", file=sys.stderr)
         sys.exit(1)
    except ValueError as ve:
        print(f"{ve}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)