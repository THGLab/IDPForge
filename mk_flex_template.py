# Auxilary file for preparing the LDR inputs with flexible domains
# created by OZ, 1/8/25
# modified by SDC, 11/1/25

import numpy as np
import mdtraj as md
import argparse
import sys
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from idpforge.utils.np_utils import process_pdb, assign_rama, get_chi_angles 

def random_rotation_matrix():
    """
    Generate a random 3D rotation matrix (uniformly sampled from SO(3)).
    """
    return Rotation.random().as_matrix()

def random_rotate_translate_coords_scaled(
    coords_centered, ref_center, d
):
    """
    Applies a random rotation and a random-direction translation to 'coords',
    then moves it to 'ref_center'.
    """
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
    """
    Tries 'max_attempts' to find a non-clashing placement for fold2 relative to fold1.
    """

    flat_fold1 = fold1_coords_rotated.reshape(-1, 3)
    flat_fold1 = flat_fold1[np.any(flat_fold1 != 0, axis=1)]
    if flat_fold1.shape[0] == 0:
        ref_center = fold1_coords_rotated[:, 1].mean(axis=0)
        return True, random_rotate_translate_coords_scaled(fold2_centered, ref_center, d_sample)
        
    fold1_tree = KDTree(flat_fold1)
    ref_center = fold1_coords_rotated[:, 1].mean(axis=0)

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
    max_outer_loops=50
):
    """
    Tries to find a non-clashing orientation for two domains.
    """
    d = 5 * idr_len ** 0.588
    
    for i in range(max_outer_loops):
        if (i + 1) % 10 == 0:
            print(f"       ... Sampling domain orientation (Outer loop {i + 1}/{max_outer_loops})", flush=True)
    
        R = random_rotation_matrix()
        
        rotated_fold1 = np.einsum('nij,jk->nik', fold1_centered, R.T)
        new_fold1 = rotated_fold1 + center_fold1
        
        success, new_fold2 = random_nonclashing_transform_scaled(
            fold2_centered, new_fold1, np.random.normal(d, d / 3), 
            max_attempts=max_attempts, 
            min_dist=min_dist
        )
            
        if success:
            return new_fold1, new_fold2 

    print(f"       Warning: sample_fold_orientation failed to find a non-clashing pose after {max_outer_loops} outer loops. Returning a *clashing* pose as fallback.", flush=True)
    fallback_d = np.random.normal(d, d / 3)
    
    R = random_rotation_matrix()
    rotated_fold1 = np.einsum('nij,jk->nik', fold1_centered, R.T)
    new_fold1 = rotated_fold1 + center_fold1
    ref_center = new_fold1[:, 1].mean(axis=0)

    fallback_fold2 = random_rotate_translate_coords_scaled(fold2_centered, ref_center, fallback_d)
    return new_fold1, fallback_fold2


def main(input_pdb, disorder_idx, nsample, **kwargs):
    """
    Main function to generate the multi-pose template .npz file.
    """
    # --- 1. Load PDB and Calculate Features ---
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

    # --- 2. Identify Domains and Linker ---
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
          
    fold1_coords = crd[:linker_start_idx]
    fold2_coords = crd[linker_end_idx + 1:]
    
    center_fold1 = fold1_coords[:, 1].mean(axis=0)
    fold1_centered = fold1_coords - center_fold1
    
    center_fold2 = fold2_coords[:, 1].mean(axis=0)
    fold2_centered = fold2_coords - center_fold2
    
    # --- 3. Generate Multiple Domain Orientations ---
    new_coords = np.tile(crd, (nsample, 1, 1, 1))
    atom_mask = new_coords.sum(axis=-1) == 0 # This is now 3D
    
    print(f"       Generating {nsample} random domain orientations (min_dist=3.8)...", flush=True)
    for i in range(nsample):
        new_fold1, new_fold2 = sample_fold_orientation(
            fold1_centered, 
            center_fold1, 
            fold2_centered,
            len(disorder_range)
        )
        
        new_coords[i, :linker_start_idx] = new_fold1
        new_coords[i, linker_end_idx + 1:] = new_fold2
        new_coords[i, linker_start_idx: linker_end_idx + 1] = 0

    center = new_coords[:, [linker_start_idx, linker_end_idx + 1], 1].mean(axis=1) + np.random.uniform(0, len(disorder_range) / 5, size=(nsample, 3))
    new_coords -= center[:, None, None, :]
    
    i_ns, i_res, i_atom = np.where(atom_mask)
    new_coords[i_ns, i_res, i_atom] = 0

    # --- 4. Build the final .npz template ---
    template = {"coord": new_coords, "torsion": torsion_vec, "sec": encode, "seq": seq}
    
    mask = np.ones(len(seq), dtype=bool)
    mask[disorder_range] = False # Set linker region to flexible
    template["mask"] = mask
    
    return template

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a multi-pose .npz template for flexible linker modeling.")
    parser.add_argument('input', help="Input PDB file path.")
    parser.add_argument('disorder_domain', help="Specify residue number for disordered region (1-index, both ends inclusive). Example: 38-129")
    parser.add_argument('output', help="Output .npz file path.")
    parser.add_argument('--nconf', default=200, type=int, help="Number of random domain orientations to sample.")
    
    args = parser.parse_args()

    try:
        i, j = args.disorder_domain.split("-")
        disorder_range = range(int(i)-1, int(j)) 
        
        # main() now handles validation
        out = main(args.input, disorder_range, args.nconf) 
        
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