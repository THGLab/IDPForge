"""Back-calculation of experimental observables from PDB ensembles."""
import os
import sys
import subprocess
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_dihedral
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

# CSpred (UCBShift) runs in its own env as a subprocess: CSPRED_PYTHON=that env's interpreter, CSPRED_PATH=CSpred.py.
CSpred_path = os.environ.get("CSPRED_PATH",
    os.path.join(os.path.dirname(__file__), '..', '..', 'Scoring', 'CSpred', 'CSpred.py'))
CSPRED_PYTHON = os.environ.get("CSPRED_PYTHON", "python")
# External binary: mkdssp (DSSP) for secondary structure.
DSSP_path = "mkdssp"


def run_command(cmd, raise_error=True, input=None, timeout=600, **kwargs):
    if isinstance(cmd, str):
        cmd = cmd.split()
    cmd = [str(x) for x in cmd]
    sub = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, **kwargs)
    if input is not None:
        sub.stdin.write(bytes(input, encoding=sys.stdin.encoding))
    try:
        out, err = sub.communicate(timeout=timeout)
        return_code = sub.poll()
    except subprocess.TimeoutExpired:
        sub.kill()
        print("Command %s timeout after %d seconds" % (cmd, timeout))
        return 999, "", ""
    out, err = out.decode(sys.stdin.encoding), err.decode(sys.stdin.encoding)
    if raise_error and return_code != 0:
        raise RuntimeError("Command %s failed: \n%s" % (" ".join(cmd), err))
    return return_code, out, err


dssp_mapping = {"B": "E", "G": "H", "I": "H", "S": "T", "-": "C"}
hydrogen_abbrev = {
    "ALA": {"HB": ["HB1", "HB2", "HB3"]},
    "ARG": {"HB": ["HB2", "HB3"], "HG": ["HG2", "HG3"], "HD": ["CD2", "CD3"],
            "HH": ["HH11", "HH12", "HH21", "HH22"], "HH1": ["HH11", "HH12"], "HH2": ["HH21", "HH22"]},
    "ASP": {"HB": ["HB2", "HB3"]},
    "ASN": {"HB": ["HB2", "HB3"], "HD": ["HD21", "HD22"], "HD2": ["HD21", "HD22"]},
    "CYS": {"HB": ["HB2", "HB3"]},
    "GLU": {"HB": ["HB2", "HB3"], "HG": ["HG2", "HG3"]},
    "GLN": {"HB": ["HB2", "HB3"], "HG": ["HG2", "HG3"], "HE": ["HE21", "HE22"], "HE2": ["HE21", "HE22"]},
    "GLY": {"HA": ["HA2", "HA3"]},
    "HIS": {"HB": ["HB2", "HB3"], "HD": ["HD2"], "HE": ["HE1"]},
    "ILE": {"HB": ["HB2", "HB3"], "HG": ["HG12", "HG13", "HG21", "HG22", "HG23"],
            "HG1": ["HG12", "HG13"], "HG2": ["HG21", "HG22", "HG23"],
            "HD": ["HD11", "HD12", "HD13"], "HD1": ["HD11", "HD12", "HD13"]},
    "LEU": {"HB": ["HB2", "HB3"], "HD": ["HD11", "HD12", "HD13", "HD21", "HD22", "HD23"],
            "HD1": ["HD11", "HD12", "HD13"], "HD2": ["HD21", "HD22", "HD23"]},
    "LYS": {"HB": ["HB2", "HB3"], "HG": ["HG2", "HG3"], "HD": ["HD2", "HD3"],
            "HZ": ["HZ1", "HZ2", "HZ3"], "HE": ["HE2", "HE3"]},
    "MET": {"HB": ["HB2", "HB3"], "HG": ["HG2", "HG3"], "HE": ["HE1", "HE2", "HE3"]},
    "PHE": {"HB": ["HB2", "HB3"], "HD": ["HD1", "HD2"], "HE": ["HE1", "HE2"]},
    "PRO": {"HB": ["HB2", "HB3"], "HG": ["HG2", "HG3"], "HD": ["HD2", "HD3"]},
    "SER": {"HB": ["HB2", "HB3"]},
    "THR": {"HB": ["HB2", "HB3"], "HG": ["HG1", "HG21", "HG22", "HG23"], "HG2": ["HG21", "HG22", "HG23"]},
    "TRP": {"HB": ["HB2", "HB3"], "HD": ["HD1"], "HZ": ["HZ2", "HZ3"], "HE": ["HE1", "HE3"], "HH": ["HH2"]},
    "TYR": {"HB": ["HB2", "HB3"], "HD": ["HD1", "HD2"], "HE": ["HE1", "HE2"]},
    "VAL": {"HG": ["HG11", "HG12", "HG13", "HG21", "HG22", "HG23"],
            "HG1": ["HG11", "HG12", "HG13"], "HG2": ["HG21", "HG22", "HG23"]},
}
heavy_atom_hydrogen = {
    "ALA": {"H": ["N"], "HA": ["CA"], "HB": ["CB"]},
    "ARG": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HG": ["CG"], "HD": ["CD"], "HE": ["NE"], "HH": ["NH1", "NH2"]},
    "ASP": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HD": ["OD1", "ND2"]},
    "ASN": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HD": ["OD1", "ND2"]},
    "CYS": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HG": ["SG"]},
    "GLU": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HG": ["CG"]},
    "GLN": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HG": ["CG"], "HE": ["OE1", "NE2"]},
    "GLY": {"H": ["N"], "HA": ["CA"]},
    "HIS": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HD": ["CD2"], "HE": ["CE1"]},
    "ILE": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HG": ["CG1", "CG2"], "HD": ["CD1"]},
    "LEU": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HG": ["CG"], "HD": ["CD1", "CD2"]},
    "LYS": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HG": ["CG"], "HD": ["CD"], "HZ": ["NZ"], "HE": ["CE"]},
    "MET": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HG": ["CG"], "HE": ["CE"]},
    "PHE": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HD": ["CD1", "CD2"], "HE": ["CE1", "CE2"], "HZ": ["CZ"]},
    "PRO": {"H": ["N"], "HB": ["CB"], "HG": ["CG"], "HD": ["CD"]},
    "SER": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HG": ["OG"]},
    "THR": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HG": ["OG1", "CG2"]},
    "TRP": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HD": ["CD1"], "HZ": ["CZ2", "CZ3"], "HE": ["NE1", "CE3"], "HH": ["CH2"]},
    "TYR": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HD": ["CD1", "CD2"], "HE": ["CE1", "CE2"], "HH": ["OH"]},
    "VAL": {"H": ["N"], "HA": ["CA"], "HB": ["CB"], "HG": ["CG1", "CG2"]},
}
atomic_mass = {"C": 12, "O": 16, "N": 14, "S": 32, "H": 1}


def split_pdbs(traj, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(traj, "r") as f:
        models = f.read().split("ENDMDL\n")
    for i, m in enumerate(models[:-1]):
        with open(os.path.join(out_dir, f"{i}.pdb"), "w") as f:
            f.write(m[12:])
    return [os.path.join(out_dir, f"{i}.pdb") for i in range(len(models[:-1]))]


def jc_backcalc(exp_data, pdbs, **kwargs):
    exp_data = pd.read_csv(exp_data)
    resn = exp_data.resnum.values
    parser = PDBParser()
    if isinstance(pdbs, str):
        pdbs = parser.get_structure('d', pdbs)
    jc_bc = np.zeros((len(pdbs), len(resn)))
    for i, pdb in enumerate(pdbs):
        struct = parser.get_structure('d', pdb)[0] if isinstance(pdb, str) else pdb
        for j, r in enumerate(resn):
            phi = calc_dihedral(struct['A'][int(r - 1)]['C'].get_vector(),
                                struct['A'][int(r)]['N'].get_vector(),
                                struct['A'][int(r)]['CA'].get_vector(),
                                struct['A'][int(r)]['C'].get_vector())
            jc_bc[i, j] = np.cos(phi - np.radians(60))
    return jc_bc


def fret_backcalc(exp_data, pdbs, **kwargs):
    exp_data = pd.read_csv(exp_data)
    res1 = exp_data.res1.values.astype(int)
    res2 = exp_data.res2.values.astype(int)
    scaler = exp_data.scale.values
    parser = PDBParser()
    if isinstance(pdbs, str):
        pdbs = parser.get_structure('d', pdbs)
    fret_bc = np.zeros((len(pdbs), len(exp_data)))
    for i, pdb in enumerate(pdbs):
        struct = parser.get_structure('d', pdb)[0] if isinstance(pdb, str) else pdb
        for j in range(exp_data.shape[0]):
            d = struct['A'][int(res1[j])]['CA'] - struct['A'][int(res2[j])]['CA']
            fret_bc[i, j] = 1.0 / (1.0 + (d / scaler[j]) ** 6.0)
    return fret_bc


def run_cspred(pdb, exp_data, tmp_outpath):
    exp = pd.read_csv(exp_data)
    cs = np.zeros(len(exp))
    with tempfile.TemporaryDirectory() as worker_tmp:
        csv = os.path.join(worker_tmp, os.path.basename(pdb).replace(".pdb", ".csv"))
        try:
            run_command([CSPRED_PYTHON, CSpred_path, pdb, "-o", csv, "-x"], cwd=worker_tmp)
        except RuntimeError as e:
            raise RuntimeError(f"CSpred failed for {pdb}") from e
        df = pd.read_csv(csv)
        for j, row in exp.iterrows():
            cs[j] = df.loc[int(row.resnum) - 1, row.atomname + "_X"]
    return cs


def cshift_backcalc(exp_data, pdbs, tmp_outpath="/tmp", **kwargs):
    if isinstance(pdbs, str):
        pdbs = split_pdbs(pdbs, out_dir=f"/tmp/{os.path.basename(pdbs)[:-4]}")
    shifts = [None] * len(pdbs)
    with ProcessPoolExecutor(max_workers=8) as pool:
        future_to_idx = {pool.submit(run_cspred, p, exp_data, tmp_outpath): i for i, p in enumerate(pdbs)}
        for future in tqdm(as_completed(future_to_idx), total=len(pdbs), desc="CS back-calc"):
            shifts[future_to_idx[future]] = future.result()
    return np.array(shifts)


def _dist_backcalc(exp_data, pdbs, heavy_atom_substitute, average, **kwargs):
    exp_data = pd.read_csv(exp_data)
    res1 = exp_data.res1.values.astype(int)
    atom1_name = exp_data.atom1.values
    res2 = exp_data.res2.values.astype(int)
    atom2_name = exp_data.atom2.values
    parser = PDBParser()
    if isinstance(pdbs, str):
        pdbs = parser.get_structure('d', pdbs)
    out = np.zeros((len(pdbs), len(exp_data)))
    for i, pdb in enumerate(pdbs):
        struct = parser.get_structure('d', pdb)[0] if isinstance(pdb, str) else pdb
        for j in range(exp_data.shape[0]):
            r1, r2 = int(res1[j]), int(res2[j])
            resn1 = struct['A'][r1].get_resname()
            resn2 = struct['A'][r2].get_resname()
            if heavy_atom_substitute:
                if atom1_name[j].startswith("H"):
                    a1 = [struct['A'][r1][a] for a in heavy_atom_hydrogen[resn1][atom1_name[j][:2]]]
                else:
                    a1 = [struct['A'][r1][atom1_name[j]]]
                if atom2_name[j].startswith("H"):
                    a2 = [struct['A'][r2][a] for a in heavy_atom_hydrogen[resn2][atom2_name[j][:2]]]
                else:
                    a2 = [struct['A'][r2][atom2_name[j]]]
            else:
                try:
                    a1 = [struct['A'][r1][atom1_name[j]]]
                except KeyError:
                    a1 = [struct['A'][r1][a] for a in hydrogen_abbrev[resn1][atom1_name[j]]]
                try:
                    a2 = [struct['A'][r2][atom2_name[j]]]
                except KeyError:
                    a2 = [struct['A'][r2][a] for a in hydrogen_abbrev[resn2][atom2_name[j]]]
            if average:  # NOE: <r^-6> over multiple assignments
                combos, num = 0.0, 0
                for x in a1:
                    for y in a2:
                        combos += (x - y) ** (-6.)
                        num += 1
                out[i, j] = (combos / float(num)) ** (-1 / 6)
            else:  # PRE: first atom only
                out[i, j] = a1[0] - a2[0]
    return out


def pre_backcalc(exp_data, pdbs, heavy_atom_substitute=False, **kwargs):
    return _dist_backcalc(exp_data, pdbs, heavy_atom_substitute, average=False, **kwargs)


def noe_backcalc(exp_data, pdbs, heavy_atom_substitute=False, **kwargs):
    return _dist_backcalc(exp_data, pdbs, heavy_atom_substitute, average=True, **kwargs)


def calc_dssp(pdb):
    dssp_dict = dssp_dict_from_pdb_file(pdb, DSSP_path)
    dssp = "".join([dssp_dict[0][key][1] for key in dssp_dict[1]])
    for k, v in dssp_mapping.items():
        dssp = dssp.replace(k, v)
    return dssp


def calc_rg(pdb):
    struct = PDBParser().get_structure('d', pdb)
    rgs = []
    for model in struct:
        atom_mass, coords, cm = [], [], np.zeros(3)
        for res in model["A"]:
            for atom in res:
                atom_mass.append(atomic_mass[atom.element])
                coords.append(atom.get_coord())
                cm += atom.get_coord() * atomic_mass[atom.element]
        cm /= np.sum(atom_mass)
        rgs.append(np.sqrt(np.sum((np.array(coords) - cm[None, :]) ** 2 * np.array(atom_mass)[:, None]) / np.sum(atom_mass)))
    return rgs[0] if len(rgs) == 1 else np.array(rgs)


BACK_Calculators = {
    "jc": jc_backcalc, "fret": fret_backcalc, "noe": noe_backcalc,
    "pre": pre_backcalc, "cs": cshift_backcalc,
}
