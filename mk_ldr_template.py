# Auxilary file for preparing the LDR inputs
# created by OZ, 1/8/25

import numpy as np
import mdtraj as md
from idpforge.utils.np_utils import (
    process_pdb, assign_rama,
    get_chi_angles
)


def sample_fix_distance(batch_size, fixed_distance, exclude_center=[], exclude_radius=[]):
    # Generate random angles
    theta = np.random.uniform(0, 2 * np.pi, batch_size)  # Azimuthal angle
    phi = np.random.uniform(0, np.pi, batch_size)        # Polar angle

    x = fixed_distance * np.sin(phi) * np.cos(theta)
    y = fixed_distance * np.sin(phi) * np.sin(theta)
    z = fixed_distance * np.cos(phi)
    vectors = np.stack([x, y, z], axis=1)
    # Exclude inits that are within exclude radius to the center
    for c, r in zip(exclude_center, exclude_radius):
        distances = np.linalg.norm(vectors - c, axis=1)
        vectors = vectors[distances > r]
    return vectors

def est_distance(n, min_clip=10):
    return max(0.03*n**2 - 0.62*n + 23, min_clip)

def calc_rg(crd):
    return np.sqrt(np.sum((crd - crd.mean(axis=0))**2, axis=-1)).mean(axis=0)


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

    if disorder_idx[0] == 0:
        folded_center = crd[disorder_idx[-1]+1, 1]
        exclude_center = [crd[disorder_idx[-1]+1:, 1].mean(axis=0)]
        exclude_radius = [calc_rg(crd[disorder_idx[-1]+1:, 1])]
    elif disorder_idx[-1] == crd.shape[0] - 1:
        folded_center = crd[disorder_idx[0]-1, 1]
        exclude_center = [crd[:disorder_idx[0], 1].mean(axis=0)]
        exclude_radius = [calc_rg(crd[:disorder_idx[0], 1])]
    else:
        folded_center = crd[[disorder_idx[0]-1, disorder_idx[-1]+1], 1].mean(axis=0)
        exclude_center = [crd[:disorder_idx[0], 1].mean(axis=0), crd[disorder_idx[-1]+1:, 1].mean(axis=0)]
        exclude_radius = [[calc_rg(crd[:disorder_idx[0], 1]), calc_rg(crd[disorder_idx[-1]+1:, 1])]
    crd -= folded_center
    d = est_distance(len(disorder_idx), **kwargs)
    noise_init = sample_fix_distance(nsample, d, exclude_center, [x * 0.8 for x in exclude_radius])
    while len(noise_init) < nsample:
        noise_init = np.concatenate((noise_init, sample_fix_distance(nsample, d, exclude_center, [x * 0.8 for x in exclude_radius])))
    permuted = noise_init[:nsample] + np.random.normal(0, d / 5, size=(nsample, 3))

    i, j = np.where(atom_mask)
    crd[i, j] = 0
    template = {"coord": crd, "torsion": torsion_vec, "sec": encode, "seq": seq}

    mask = np.ones(len(crd), dtype=bool)
    mask[disorder_idx] = False
    template["mask"] = mask
    template["coord_offset"] = permuted
    return template


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('disorder_domain', help="Specify residue number for disordered region (1-index, both ends inclusive). Current model is not optimized for multiple domains. Example: 1-15 or 38-129")
    parser.add_argument('output', help=".npz")
    parser.add_argument('--nconf', default=200, type=int)
    args = parser.parse_args()

    i, j = args.disorder_domain.split("-")
    out = main(args.input, range(int(i)-1, int(j)), args.nconf)
    np.savez(args.output, **out)
