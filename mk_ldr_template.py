# Auxilary file for preparing the LDR inputs
# created by OZ, 1/8/25
# modified by SDC, 11/1/25

import numpy as np
import mdtraj as md
import argparse
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


def sample_fix_distance(batch_size, fixed_distance, 
                        exclude_center=[], exclude_radius=[], tol=0.2): # Default: 0.8 Rg
    """
    Generates points on a sphere, vectorized to efficiently exclude multiple
    regions.
    """
    if batch_size == 0:
        return np.empty((0, 3))

    # Generate random angles
    theta = np.random.uniform(0, 2 * np.pi, batch_size)
    phi = np.random.uniform(0, np.pi, batch_size)

    # Convert to Cartesian coordinates
    x = fixed_distance * np.sin(phi) * np.cos(theta)
    y = fixed_distance * np.sin(phi) * np.sin(theta)
    z = fixed_distance * np.cos(phi)
    vectors = np.stack([x, y, z], axis=1)

    # If there's nothing to exclude, return all vectors
    if not exclude_center:
        return vectors

    # Convert inputs to numpy arrays for broadcasting
    centers = np.atleast_2d(np.array(exclude_center))

    # Calculate squared radii for comparison
    radii_sq = (np.array(exclude_radius) * tol)**2 

    # Calculate differences between each vector and each center
    diffs = vectors[:, np.newaxis, :] - centers[np.newaxis, :, :]

    # Calculate squared distances
    dists_sq = np.einsum('nmj,nmj->nm', diffs, diffs)

    # Check if distances are less than squared radii
    is_too_close = dists_sq < radii_sq[np.newaxis, :]

    # Identify points that are too close to any exclusion zone
    is_bad_point = np.any(is_too_close, axis=1)

    # Return only the 'good' points
    return vectors[~is_bad_point]


def est_distance(n, min_clip=12):
    """Estimate distance based on sequence length."""
    return max(2.4 * n ** 0.588, min_clip)


def calc_rg(crd):
    """Optimized Rg calculation."""
    # Filter out 0,0,0 atoms before calculating Rg
    valid_atoms = crd[np.any(crd != 0, axis=1)]
    if valid_atoms.shape[0] == 0:
        return 1.0 # Return a small default if no valid atoms
    
    # Calculate deviations from the mean
    mean_coord = valid_atoms.mean(axis=0)
    dev = valid_atoms - mean_coord
    
    # Calculate Rg^2
    rg_sq = np.einsum('ij,ij->', dev, dev) / valid_atoms.shape[0]
    return np.sqrt(rg_sq)


def main(pdb, disorder_idx, nsample, **kwargs):
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
    atom_mask = crd.sum(axis=-1) == 0

    disorder_idx_list = list(disorder_idx)
    if not disorder_idx_list:
         raise ValueError("Disorder index list is empty.")
         
    start_idx = disorder_idx_list[0]
    end_idx = disorder_idx_list[-1]

    # --- Define anchor point and exclusion zones ---
    if start_idx == 0: # N-terminal tail
        folded_center = crd[end_idx+1, 1]
        exclude_coords = [crd[end_idx+1:, 1]]
    elif end_idx == crd.shape[0] - 1: # C-terminal tail
        folded_center = crd[start_idx-1, 1]
        exclude_coords = [crd[:start_idx, 1]]
    else: # Loop
        folded_center = crd[[start_idx-1, end_idx+1], 1].mean(axis=0)
        exclude_coords = [crd[:start_idx, 1], crd[end_idx+1:, 1]]
    
    # Calculate COM and Rg for exclusion zones
    exclude_center = [c.mean(axis=0) for c in exclude_coords if c.shape[0] > 0]
    exclude_radius = [calc_rg(c) for c in exclude_coords if c.shape[0] > 0]
    
    # 1. Shift main coordinates so anchor point is at (0,0,0)
    crd -= folded_center
    # 2. Shift exclusion centers relative to the new origin
    shifted_exclude_center = [c - folded_center for c in exclude_center]
    
    d = est_distance(len(disorder_idx_list), **kwargs)
    
    batch_multiplier = 10 # Start by sampling 10x more than needed
    max_tries = 10
    tries = 0
    
    # Initial attempt
    initial_batch_size = nsample * batch_multiplier
    noise_init_list = [sample_fix_distance(initial_batch_size, d, 
                                           shifted_exclude_center, exclude_radius)]
    
    points_found = len(noise_init_list[0])

    while points_found < nsample and tries < max_tries:
        print(f"     ... {points_found}/{nsample} points found. Sampling more (Try {tries + 1}/{max_tries})", flush=True)
        
        needed = nsample - points_found

        # Request a new batch, scaled by the multiplier
        batch_size = needed * batch_multiplier
        
        new_points = sample_fix_distance(batch_size, d, 
                                          shifted_exclude_center, exclude_radius)
        
        if len(new_points) == 0:
            # If we get nothing, the region is tough to sample.
            # Increase multiplier for the next attempt.
            batch_multiplier *= 2
        else:
            noise_init_list.append(new_points)
            points_found += len(new_points)
        
        tries += 1

    # --- Fallback and Finalizing ---
    if points_found == 0:
        print(f"     Warning: Could not find any non-clashing points. Using a default vector. It is recommended to review this template.", flush=True)
        noise_init = np.array([[d, 0.0, 0.0]]) # A simple default point
    else:
        noise_init = np.concatenate(noise_init_list, axis=0)
         
    # Ensure the final array has the correct shape
    if len(noise_init) < nsample:
        print(f"     Warning: Only found {len(noise_init)}/{nsample} points. Duplicating to fill. It is recommended to review this template.", flush=True)
        repeat_times = int(np.ceil(nsample / len(noise_init)))
        noise_init = np.tile(noise_init, (repeat_times, 1))
    
    # Safely take the first nsample points and add noise
    permuted = noise_init[:nsample] + np.random.normal(0, 3, size=(nsample, 3))

    # Zero-out missing atoms
    i, j = np.where(atom_mask)
    crd[i, j] = 0
    
    # Create template
    template = {"coord": crd, "torsion": torsion_vec, "sec": encode, "seq": seq}

    mask = np.ones(len(crd), dtype=bool)
    mask[disorder_idx_list] = False
    template["mask"] = mask
    template["coord_offset"] = permuted
    return template


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare LDR inputs from a PDB file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', help="Input PDB file.")
    parser.add_argument('disorder_domain', help="Specify residue number for disordered region (1-index, both ends inclusive). Example: 1-15 or 38-129")
    parser.add_argument('output', help="Output .npz file.")
    parser.add_argument('--nconf', default=200, type=int, help="Number of conformations (starting points) to generate.")
    args = parser.parse_args()

    try:
        i, j = args.disorder_domain.split("-")
        # Convert from 1-based (inclusive) to 0-based (exclusive end)
        disorder_range = range(int(i)-1, int(j)) 
        out = main(args.input, disorder_range, args.nconf)
        np.savez(args.output, **out)
        print(f"Successfully generated {args.output} with {args.nconf} conformations.")
    except Exception as e:
        print(f"Error processing {args.input} with {args.disorder_domain}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)