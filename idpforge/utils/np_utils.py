import numpy as np
import pandas as pd
import re
import random
from openfold.np.residue_constants import (
    chi_angles_atoms, 
    restype_1to3, 
    restype_3to1,
    restypes,
    restype_name_to_atom14_names, 
)
from idpforge.utils.definitions import (
    sstypes, coil_sample_probs
)

def assign_rama(bbs):
    # encode rama designation for alpha, beta and coil region
    encode = ""
    phis = np.degrees(bbs[:, 0])
    psis = np.degrees(bbs[:, 1])
    for phi, psi in zip(phis, psis):
        if (-120.0 < phi < -20.0) and (-100.0 < psi < 60.0):
            encode += "A"
        elif (-180.0 <= phi < -100.0 or 150 < phi <= 180) and ((-180 <= psi < -120) or (60 < psi <= 180)):
            encode += "B"
        elif (30.0 < phi < 100.0) and (-50.0 < psi < 100.0):
            encode += "L"
        elif (-100.0 <= phi < -20) and ((-180 <= psi < -120) or (60 < psi <= 180)):
            encode += "P"
        else:
            encode += "C"
    return encode


def calc_rg(coords):
    # backbone based
    mass = np.array([[14, 12, 12, 16]] * len(coords))
    center = np.sum(coords * mass[..., None], axis=(0, 1)) / mass.sum()
    return np.sqrt(np.sum((coords - center[None, None, :])**2 * mass[..., None]) / mass.sum())


def get_jcoup_array(exp, nseq, rand_threshold=1):
    df = pd.read_csv(exp, index_col=0)
    jcoup = np.zeros(nseq - 1)
    for i, row in df.iterrows():
        if np.random.random() < rand_threshold:
            jcoup[int(row.resnum - 2)] = row.value
    return jcoup

def get_contact_map(exp, nseq, rand_threshold=1):
    df = pd.read_csv(exp, index_col=0)
    cmap = np.zeros((nseq, nseq, 2))
    for i, row in df.iterrows():
        if np.random.random() < rand_threshold:
            cmap[int(row.res1 - 1), int(row.res2 - 1), 0] = row.dist_value - row.lower 
            cmap[int(row.res2 - 1), int(row.res1 - 1), 0] = row.dist_value - row.lower
            cmap[int(row.res1 - 1), int(row.res2 - 1), 1] = row.dist_value + row.upper
            cmap[int(row.res2 - 1), int(row.res1 - 1), 1] = row.dist_value + row.upper
    return cmap

def get_efret_array(exp, nseq, rand_threshold=1):
    df = pd.read_csv(exp, index_col=0)
    eff = np.zeros((nseq, nseq, 2))
    eff[..., 1] = 1
    for i, row in df.iterrows():
        if np.random.random() < rand_threshold:
            eff[int(row.res1 - 1), int(row.res2 - 1), 0] = row.value
            eff[int(row.res2 - 1), int(row.res1 - 1), 0] = row.value
            eff[int(row.res1 - 1), int(row.res2 - 1), 1] = row.scale
            eff[int(row.res2 - 1), int(row.res1 - 1), 1] = row.scale
    return eff


def read_d2d(d2d):
    dict_out = {}
    dssp_keys = ["A", "E", "C", "P"]

    with open(d2d, "r") as reader:
        for line in reader:
            dict_probs = {}
            if "#" not in line and line != "\n":
                pline = line.split()
                pline.pop(1)
                pline.pop()
                data = [float(i) for i in pline]
                resid = int(data[0])
                data.pop(0)
                if np.sum(data) != 1:
                    # make sure all prob sums to 1 for random choice
                    data[-2] = 1 - np.sum(data[:2]) - data[-1]
                for i, prob in enumerate(data):
                    dict_probs[dssp_keys[i]] = prob
                dict_out[resid] = dict_probs
            # if there aren't any predicted probabilities
            # for this residue, make them all loop
            elif re.match(r"\#+\d", line):
                sline = line.split()
                pline = sline[0].split("#")
                resid = int(pline[1].strip())
                dict_out[resid] = {"A": 0.25, "E": 0.25, "C": 0.25, "P": 0.25}
    return dict_out
        
    
def sample_ss(l, d2d=None, ss_probs=None):
    if ss_probs is None:
        ss_probs = randomize_ss_probs()
    if isinstance(list(ss_probs.keys())[0], str):
        ss_probs = {i+1: ss_probs for i in range(l)}
                    
    ss = ""
    i = 1
    while i <= l:
        if d2d is not None and i in d2d:
            letters = list(d2d[i].keys())
            probs = list(d2d[i].values())
        else:
            letters = list(ss_probs[i].keys()) 
            probs = [v for k, v in ss_probs[i].items()]
        s = np.random.choice(letters, p=probs)
        if s == "E":
            ss += "EE"
            i += 2
        elif s == "A":
            if d2d is not None or random.random() > 0.5:
                ss += "HHHHH"
                i += 5
            elif random.random() < 0.2:
                ss += "HHH"
                i += 3
            else:
                ss += "HHHH"
                i += 4
        elif s == "C" and d2d is not None:
            ss += "C" if random.random() < 0.4 else "P"
            i += 1
        else:
            ss += s
            i += 1
            
    if len(ss) > l:
        return ss[:l]

    if ss.count("EE") == 1:
        ss_idx = ss.index("EE")
        c_idx = [ss_idx+i for i in range(-9, -3) if ss_idx+i >= 0 and "H" not in ss[ss_idx+i:ss_idx+i+2]]
        c_idx += [ss_idx+i for i in range(4, 10) if ss_idx+i < l-1 and "H" not in ss[ss_idx+i:ss_idx+i+2]]
        if len(c_idx) > 0:
            random.shuffle(c_idx)
            ss = ss[:c_idx[0]] + "EE" + ss[c_idx[0]+2:]
    return ss


def randomize_ss_probs(seed=None):
    if seed:
        np.random.seed(seed)
    random_points = np.random.rand(3)
    random_points.sort()
    norm_probs = np.diff([0] + list(random_points) + [1])
    norm_probs.sort()
    offset = np.minimum(0.15, norm_probs[0]/2)
    norm_probs[0] -= offset
    norm_probs[-2] += offset
    offset = np.minimum(0.15, norm_probs[1]/2)
    norm_probs[1] -= offset
    norm_probs[-1] += offset
    return {k: v for k, v in zip(["E", "A", "P", "C"], norm_probs)}

def coord_to_pdb(coords: np.ndarray, sequence: np.ndarray, mask=None) -> str:
    """
    coords nres x 9 x 3 [N, CA, C, O, CB ...]
    """
    # crop coordinates for padded sequences
    assert len(coords) == len(sequence)
    if mask is not None:
        coords = coords[mask]
        sequence = sequence[mask]
    if isinstance(sequence, str):
        resn3l = [restype_1to3[s] for s in sequence]
    else:
        resn3l = [restype_1to3[restypes[s]] for s in sequence]
    pdb = ""
    k = 1
    for resi, resn in enumerate(resn3l):
        aname = restype_name_to_atom14_names[resn]
        j = 0
        while j < coords.shape[1] and len(aname[j]) > 0:
            pdb += "ATOM  %5d  %-3s %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n" % (
                    k, aname[j], resn, "A", resi + 1,
                    coords[resi, j, 0], coords[resi, j, 1], coords[resi, j, 2], 1, 0)
            j += 1
            k += 1
    return pdb

def rigid_from_3_points_np(xyz, eps=1e-8):
    # N, Ca, C - [B, L, 3]
    # R - [B, L, 3, 3], det(R)=1, inv(R) = R.T, R is a rotation matrix
    N = xyz[..., 0, :]
    Ca = xyz[..., 1, :]  # [1, L, 3, 3]
    C = xyz[..., 2, :]
    B, L = N.shape[:2]

    v1 = C - Ca
    v2 = N - Ca

    e1 = v1 / (np.linalg.norm(v1, axis=-1)[..., None] + eps)
    u2 = v2 - (np.einsum("bli, bli -> bl", e1, v2)[..., None] * e1)
    e2 = u2 / (np.linalg.norm(u2, axis=-1)[..., None] + eps)
    e3 = np.cross(e1, e2, axis=-1)
    R = np.stack([e1, e2, e3], axis=-1)  # [B,L,3,3] - rotation matrix
    return R

def get_dih_np(a, b, c, d):
    """calculate dihedral angles for all consecutive quadruples (a[i], b[i], c[i], d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors or numpy array of shape [batch, nres, 3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor or numpy array of shape [batch, nres]
          stores resulting dihedrals
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)

    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)

    return np.arctan2(y, x)

def get_chi_angles(xyz, sequence):
    """
    Get chi angles for a given sequence
    :param xyz: (N, 9, 3) array of coordinates
    :param sequence: string of amino acid sequence
    :return: (N, 4) array of chi angles
    """
    chi_angles = np.ones((xyz.shape[0], 4)) * -np.pi
    if isinstance(sequence, str):
        ile_mask = np.array([res == "I" for res in sequence])
        sidechain_mask = get_chi_mask_np(sequence)
    else:
        ile_mask = sequence == restypes.index("I")
        sidechain_mask = get_chi_mask_np(sequence)
    chi1_mask = sidechain_mask[:, 0]
    chi2_mask = sidechain_mask[:, 1]
    chi3_mask = sidechain_mask[:, 2]
    chi4_mask = sidechain_mask[:, 3]

    chi_angles[chi1_mask, 0] = get_dih_np(xyz[chi1_mask, 0], xyz[chi1_mask, 1],
        xyz[chi1_mask, 4], xyz[chi1_mask, 5])
    if chi2_mask.sum():
        chi_angles[chi2_mask, 1] = get_dih_np(xyz[chi2_mask, 1], xyz[chi2_mask, 4],
            xyz[chi2_mask, 5], xyz[chi2_mask, 6])
    if chi3_mask.sum():
        chi_angles[chi3_mask, 2] = get_dih_np(xyz[chi3_mask, 4], xyz[chi3_mask, 5],
            xyz[chi3_mask, 6], xyz[chi3_mask, 7])
    if chi4_mask.sum():
        chi_angles[chi4_mask, 3] = get_dih_np(xyz[chi4_mask, 5], xyz[chi4_mask, 6],
            xyz[chi4_mask, 7], xyz[chi4_mask, 8])
    if ile_mask.sum():
        chi_angles[ile_mask, 1] = get_dih_np(xyz[ile_mask, 1], xyz[ile_mask, 4],
            xyz[ile_mask, 5], xyz[ile_mask, 7])

    return chi_angles, np.stack([chi1_mask, chi2_mask, chi3_mask, chi4_mask], axis=1)

def get_chi_mask_np(sequence):
    if isinstance(sequence, str):
        chi_num = np.array([len(chi_angles_atoms[restype_1to3[res]]) for res in sequence])
    else:
        chi_num = np.array([len(chi_angles_atoms[restypes[res]]) for res in sequence])
    chi1_mask = chi_num > 0
    chi2_mask = chi_num > 1
    chi3_mask = chi_num > 2
    chi4_mask = chi_num > 3

    return np.stack([chi1_mask, chi2_mask, chi3_mask, chi4_mask], axis=1)


def reorder_atoms(atoms, resi, seq):
    s = []
    max_atom = 0
    offset = resi[0]
    for i in np.unique(resi):
        chunk = atoms[resi == i]
        nchi = np.sum(resi == i)
        try:
            template = restype_name_to_atom14_names[restype_1to3[seq[i - offset]]][:nchi]
            s += [list(chunk).index(a) + max_atom for a in template]
            max_atom += nchi
        except ValueError:
            print(seq[i - offset], i)
            print(template)
            print(chunk)
    return s

def process_pdb(path):
    with open(path, "r+") as f:
        pdblines = f.readlines()

    pdblines = [l for l in pdblines if l.startswith("ATOM") or l.startswith("HETATM")]
    resi = np.array([l[22:26].strip() for l in pdblines], dtype=int)
    resn = np.array([l[17:20].strip() for l in pdblines])
    atoms = np.array(["O" if l[12:16].strip() =="OT1" else l[12:16].strip() for l in pdblines])
    coords = np.array([[l[30:38].strip(), l[38:46].strip(), l[46:54].strip()] for l in pdblines],
                      dtype=float)
    seq = "".join([restype_3to1[r] for a, r in zip(atoms, resn) if a=="CA"])
    mask = np.array([(a.startswith("C") or a.startswith("O") or a.startswith("N") or a.startswith("S")) and \
                     a not in ["OXT", "OT2", "O1P", "O2P", "O3P"] for a in atoms])
    heavy_coords = coords[mask]
    reorder_index = reorder_atoms(atoms[mask], resi[mask], seq)
    heavy_coords = heavy_coords[reorder_index]
    new_coords = np.zeros((len(seq), 14, 3))
    for i, r in enumerate(np.unique(resi[mask])):
        new_coords[i, :np.sum(resi[mask] == r)] = heavy_coords[resi[mask] == r]
    return new_coords, seq

