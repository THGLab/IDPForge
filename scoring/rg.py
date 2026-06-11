"""Ensemble radius of gyration for the --normalize %|dRg|/Rg column.

Mass-weighted Rg over all atoms (incl. H) of chain A. Default 30x100 trials ->
rg_trials.csv (cols trial,rg, read by normalize.load_rg_trials); --all -> per-frame
rg.csv. Seed is deterministic per (method, protein).
"""
import os
import hashlib
import numpy as np
from glob import glob

# Frame-file conventions across methods, relaxed preferred. `conformer_*.pdb`
# is a last resort for unrelaxed idpconfg sets; no bare `*.pdb` catch-all
# (it would grab CALVADOS topology/trajectory PDBs).
FRAME_PATTERNS = ("frame_*.pdb", "*_validated.pdb", "*_relaxed.pdb",
                  "*_relax.pdb", "*_raw.pdb", "conformer_*.pdb")
FRAME_SUBDIRS = ("", "_relaxed", "traj_pdbs", "trajs")

# Atomic masses, matching scoring.calculator.atomic_mass.
ATOMIC_MASS = {"C": 12, "O": 16, "N": 14, "S": 32, "H": 1}


def _element_of(line):
    """Element for a PDB ATOM/HETATM line: element column (77-78) if populated,
    else the atom name's first alphabetic char. Matches Biopython's atom.element
    for the C/N/O/S/H atoms in protein ensembles."""
    elem = line[76:78].strip().upper() if len(line) >= 78 else ""
    if elem:
        return elem
    for ch in line[12:16]:
        if ch.isalpha():
            return ch.upper()
    return ""


def _atom_coords_mass(pdb_path):
    """(coords, masses), ALL atoms incl. H, chain A; falls back to all chains
    if no chain-A atoms exist."""
    ca, ma, c_all, m_all = [], [], [], []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            m = ATOMIC_MASS.get(_element_of(line))
            if m is None:
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            c_all.append(xyz); m_all.append(m)
            if line[21:22] == "A":
                ca.append(xyz); ma.append(m)
    if ca:
        return np.asarray(ca, float), np.asarray(ma, float)
    return np.asarray(c_all, float), np.asarray(m_all, float)


def calc_rg(pdb_path):
    """Mass-weighted Rg, matching scoring.calculator.calc_rg:
        cm = sum(m_i r_i)/sum(m_i);  Rg = sqrt(sum(m_i |r_i-cm|^2)/sum(m_i)).
    NaN on empty input."""
    coords, mass = _atom_coords_mass(pdb_path)
    if coords.size == 0:
        return float("nan")
    total = mass.sum()
    cm = (coords * mass[:, None]).sum(axis=0) / total
    dev = coords - cm
    return float(np.sqrt(np.sum((dev ** 2).sum(axis=1) * mass) / total))


def frames_in(prot_dir):
    for sub in FRAME_SUBDIRS:
        d = os.path.join(prot_dir, sub) if sub else prot_dir
        if not os.path.isdir(d):
            continue
        for pat in FRAME_PATTERNS:
            frames = sorted(glob(os.path.join(d, pat)))
            if frames:
                return frames
    return []


def discover(ens_base, methods, proteins):
    if not methods:
        methods = [os.path.basename(p) for p in sorted(glob(os.path.join(ens_base, "*"))) if os.path.isdir(p)]
    pairs = []
    for m in methods:
        ps = proteins or [os.path.basename(p) for p in sorted(glob(os.path.join(ens_base, m, "*"))) if os.path.isdir(p)]
        pairs.extend((m, p) for p in ps)
    return pairs


def process_one(method, protein, ens_base, all_frames, ens_size, trials, force):
    prot_dir = os.path.join(ens_base, method, protein)
    out_csv = os.path.join(prot_dir, "rg.csv" if all_frames else "rg_trials.csv")
    if not os.path.isdir(prot_dir):
        return "skip (no dir)"
    if os.path.exists(out_csv) and not force:
        return f"skip-existing ({os.path.basename(out_csv)}; use --force)"
    frames = frames_in(prot_dir)
    if not frames:
        return "skip (no PDBs)"
    rgs = np.asarray([calc_rg(f) for f in frames], dtype=float)
    n_pool = len(rgs)
    if all_frames:
        with open(out_csv, "w") as fh:
            fh.write("frame,rg\n")
            for f, r in zip(frames, rgs):
                fh.write(f"{os.path.basename(f)},{r:.4f}\n")
        return f"all     {n_pool} frames  mean={np.nanmean(rgs):.2f}"
    # deterministic per-(method, protein) seed so reruns reproduce
    rng = np.random.default_rng(int(hashlib.sha1(f"{method}/{protein}".encode()).hexdigest()[:8], 16))
    means = np.empty(trials)
    with open(out_csv, "w") as fh:
        fh.write("trial,rg\n")
        for t in range(trials):
            means[t] = float(np.nanmean(rgs[rng.integers(0, n_pool, size=ens_size)]))
            fh.write(f"{t + 1},{means[t]:.4f}\n")
    return f"trials  {trials}x{ens_size} (pool={n_pool})  mean={means.mean():.2f}"


def run(ens_base=None, methods=None, proteins=None, all_frames=False,
        ens_size=100, trials=30, force=False):
    ens_base = ens_base or os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'ensembles')
    pairs = discover(ens_base, methods, proteins)
    out = "rg.csv" if all_frames else "rg_trials.csv"
    print(f"ens-base: {ens_base}  ->  {out}  ({len(pairs)} ensembles)")
    n_ok = 0
    for m, p in pairs:
        status = process_one(m, p, ens_base, all_frames, ens_size, trials, force)
        print(f"  [{m}/{p}] {status}")
        n_ok += status.startswith(("trials", "all"))
    print(f"Done. wrote {out} for {n_ok}/{len(pairs)} ensembles.")
