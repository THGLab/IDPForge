"""X-EISD ensemble scorer for IDPForge.

Modes:
  default      Back-calculate the requested observables from a PDB ensemble and report
               per-property MAE and X-EISD log-likelihood over 30 trials of 100-conformer
               random subsamples -> scores_trials.csv (the benchmark protocol).
  --all        Score every conformer in a single pass -> scores_all.csv.
  --rg         Write per-ensemble Rg (mass-weighted, all-atom) -> rg_trials.csv for the
               %|dRg|/Rg column, covering proteins with an exp Rg target but no NMR data.
  --normalize  Build the cross-method Eq. S11 benchmark table from the per-protein score CSVs.
"""
import os
import argparse
from glob import glob
import numpy as np
np.random.seed(42)
import pandas as pd
from Bio.PDB import PDBParser

from scoring.parser import EXP_DATA_LIB, read_bc_data
from scoring.calculator import BACK_Calculators, calc_rg
from scoring.optimizer import XEISD
from scoring import normalize, rg


def validate_pdb(pdb_path):
    try:
        struct = PDBParser(QUIET=True).get_structure('v', pdb_path)[0]
        return len(list(struct['A'].get_residues())) > 0
    except Exception:
        return False


def backcalc_ensemble(output_pdbs, exp_data, data_types, **kwargs):
    return {prop: read_bc_data(BACK_Calculators[prop](exp_data[prop], output_pdbs, **kwargs), prop)
            for prop in data_types}


def save_backcalc(protein_name, data_types, abs_dir):
    exp_data = EXP_DATA_LIB[protein_name]
    pdbs = sorted(glob(abs_dir + "/*.pdb"))
    bc_data = backcalc_ensemble(pdbs, exp_data, data_types)
    for d in data_types:
        pd.DataFrame(bc_data[d].data).to_csv(abs_dir + f"/{d}_cache.csv", index=False, header=None)


def main(protein_name, data_types, abs_dir, ens_size=100, trials=30,
         all_conformers=False, force=False):
    data_types = list(data_types)
    out_csv = os.path.join(abs_dir, f"scores_{'all' if all_conformers else 'trials'}.csv")
    if os.path.exists(out_csv) and not force:
        print(f"SKIP: {out_csv} already exists (use --force to re-score)")
        return
    exp_data = EXP_DATA_LIB[protein_name]
    pdbs = glob(abs_dir + "/*_relax*.pdb")
    if not pdbs and os.path.isdir(abs_dir + "/traj_pdbs"):
        pdbs = glob(abs_dir + "/traj_pdbs/*_relax*.pdb")
    if not pdbs:
        pdbs = glob(abs_dir + "/*.pdb")
    pdbs.sort()
    bad = [p for p in pdbs if not validate_pdb(p)]
    if bad:
        print(f"WARNING: Removing {len(bad)} invalid PDB(s):")
        for p in bad:
            print(f"  {p}")
        pdbs = [p for p in pdbs if p not in set(bad)]
    print(f"Scoring {len(pdbs)} PDBs")
    bc_data = backcalc_ensemble(pdbs, exp_data, data_types)
    for stored in ["cs"]:  # cache/restore expensive back-calcs
        if stored in data_types:
            pd.DataFrame(bc_data[stored].data).to_csv(abs_dir + f"/{stored}_cache.csv", index=False, header=None)
        elif os.path.exists(abs_dir + f"/{stored}_cache.csv"):
            bc_data[stored] = read_bc_data(pd.read_csv(abs_dir + f"/{stored}_cache.csv", header=None).values, stored)
            data_types.append(stored)
    rgs = np.array([calc_rg(p) for p in pdbs])
    xeisd = XEISD(exp_data, bc_data)

    if all_conformers:
        n_pool = len(pdbs)
        print(f"Scoring all {n_pool} conformers in a single pass")
        ens_size, trials = n_pool, 1
        sampler = lambda: np.arange(n_pool)
    else:
        print(f"Scoring {trials} random subsamples of {ens_size} conformers (with replacement)")
        sampler = lambda: np.random.randint(len(pdbs), size=ens_size)

    rows = []
    for _ in range(trials):
        indices = sampler()
        results = xeisd.calc_scores(data_types, indices)
        row = {}
        for prop in data_types:
            row[prop + '_mae'] = results[prop][0]
            row[prop + '_score'] = results[prop][1]
        if "cs" in data_types:
            for atom, val in results['cs_per_atom_mae'].items():
                row[f'cs_{atom}_mae'] = val
            row['cs_chi2'] = results['cs_chi2']
            for atom, val in results['cs_per_atom_chi2'].items():
                row[f'cs_{atom}_chi2'] = val
        row['rg'] = np.mean(rgs[indices])
        rows.append(row)

    output = pd.DataFrame(rows)
    print('{0: >35} {1: >25}'.format('mean', 'stdev'))
    for col in output.columns:
        print(f'{col: <15} {output[col].mean(): >25.3f} {output[col].std(): >25.3f}')
    print(f'sample_rg       {np.mean(rgs): >25.3f} {np.std(rgs): >25.3f}')
    output.to_csv(out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="X-EISDv1 ensemble scorer. Default: 30 trials of 100-conformer random "
                    "subsamples; --all scores every conformer once; --normalize builds the "
                    "cross-method Eq. S11 benchmark table from existing scores.csv files.")
    parser.add_argument('protname', nargs='?')
    parser.add_argument('pdbpath', nargs='?')
    parser.add_argument('--jc', action="store_true")
    parser.add_argument('--cs', action="store_true")
    parser.add_argument('--noe', action="store_true")
    parser.add_argument('--pre', action="store_true")
    parser.add_argument('--fret', action="store_true")
    parser.add_argument('--all', dest='all_conformers', action='store_true',
                        help="score every conformer in one pass -> scores_all.csv (default: 30 "
                             "trials of --ens-size random subsamples -> scores_trials.csv).")
    parser.add_argument('--ens-size', type=int, default=100)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--force', action='store_true', help="re-score even if scores.csv exists.")
    parser.add_argument('--normalize', action='store_true',
                        help="cross-method Eq. S11 benchmark table from existing per-protein score CSVs.")
    parser.add_argument('--rg', action='store_true',
                        help="compute per-ensemble Rg -> rg_trials.csv (mass-weighted, all-atom; for "
                             "the --normalize %%|dRg|/Rg column; --all -> per-frame rg.csv).")
    parser.add_argument('--ens-base', help="[normalize/rg] base dir holding {method}/{protein}/ (default ../output/ensembles).")
    parser.add_argument('--score-file', default='scores_trials.csv',
                        help="[normalize] per-protein score CSV to aggregate (default scores_trials.csv).")
    parser.add_argument('--outdir', help="[normalize] output dir for tables (default ../output).")
    parser.add_argument('--rg-json', help="[normalize] exp Rg JSON for %%|dRg|/Rg column (optional).")
    parser.add_argument('--exp-dir', help="[normalize] exp data dir (default $IDPFORGE_EXP_DATA or ../Data/exp).")
    parser.add_argument('--exclude-methods', default=None,
                        help="[normalize] comma-separated method dirs to drop (e.g. 'desres' for Table 1).")
    parser.add_argument('--methods', default=None,
                        help="[rg] comma-separated method dirs to restrict to (default: all).")
    parser.add_argument('--proteins', default=None,
                        help="[normalize/rg] comma-separated protein subset to restrict to (e.g. the Table 2 MD subset).")
    args = parser.parse_args()

    if args.normalize:
        excl = args.exclude_methods.split(",") if args.exclude_methods else None
        prots = args.proteins.split(",") if args.proteins else None
        normalize.run(ens_base=args.ens_base, outdir=args.outdir, exp_dir=args.exp_dir,
                      rg_json=args.rg_json, score_file=args.score_file,
                      exclude_methods=excl, proteins=prots)
    elif args.rg:
        meths = args.methods.split(",") if args.methods else None
        prots = args.proteins.split(",") if args.proteins else None
        rg.run(ens_base=args.ens_base, methods=meths, proteins=prots,
               all_frames=args.all_conformers, ens_size=args.ens_size,
               trials=args.trials, force=args.force)
    else:
        if not args.protname or not args.pdbpath:
            parser.error("protname and pdbpath are required for scoring (or pass --normalize/--rg)")
        print(args.pdbpath)
        bool_flags = {'all_conformers', 'force', 'normalize', 'rg'}
        data_types = [a for a, v in vars(args).items()
                      if isinstance(v, bool) and v and a not in bool_flags]
        main(args.protname, data_types, args.pdbpath,
             ens_size=args.ens_size, trials=args.trials,
             all_conformers=args.all_conformers, force=args.force)
