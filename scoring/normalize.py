"""Cross-method benchmark table from per-protein score CSVs (X-EISD, Eq. S11).

Normalization (Eq. S11): X_norm = (X - X_min) / (X_max - X_min), per protein across
methods, per category. NOE and PRE are merged (summed) into one "NOE/PRE" category.
Methods are the immediate subdirectories of ens_base ({method}/{protein}/{score_file}),
keyed by directory name; score_file defaults to scores_trials.csv.
"""
import os
import json
import numpy as np
import pandas as pd
from glob import glob


def derive_canonical_sets(exp_dir):
    sets = {"CS": [], "JC": [], "NOE": [], "PRE": [], "FRET": []}
    type_to_cat = {"cs": "CS", "jc": "JC", "noe": "NOE", "pre": "PRE", "fret": "FRET"}
    if not os.path.isdir(exp_dir):
        print(f"WARNING: exp dir not found: {exp_dir}")
        return sets
    for protein in sorted(os.listdir(exp_dir)):
        pdir = os.path.join(exp_dir, protein)
        if not os.path.isdir(pdir):
            continue
        for fname in os.listdir(pdir):
            if not fname.endswith("_exp.txt"):
                continue
            parts = fname[:-len("_exp.txt")].split("_", 1)
            if len(parts) == 2 and parts[0] == protein and parts[1] in type_to_cat:
                sets[type_to_cat[parts[1]]].append(protein)
    sets["NOE/PRE"] = sorted(set(sets["NOE"]) | set(sets["PRE"]))
    return sets


def discover_methods(ens_base):
    return {os.path.basename(p): p for p in sorted(glob(os.path.join(ens_base, "*")))
            if os.path.isdir(p)}


def load_scores(method_path, score_file):
    results = {}
    for protein_dir in sorted(glob(method_path + "/*/")):
        f = os.path.join(protein_dir, score_file)
        if os.path.exists(f):
            results[os.path.basename(protein_dir.rstrip("/"))] = pd.read_csv(f, index_col=0)
    return results


def load_rg_trials(method_path, protein):
    """Mean Rg from {method}/{protein}/rg_trials.csv (columns: trial,rg), or None
    if missing/empty. Also checks the {protein}%00 dir for IDs colliding with
    OS-reserved names (e.g. 'nul')."""
    for name in (protein, protein + "%00"):
        f = os.path.join(method_path, name, "rg_trials.csv")
        if os.path.exists(f):
            s = pd.read_csv(f)
            return float(s["rg"].mean()) if "rg" in s.columns and len(s) else None
    return None


def normalize_per_protein(table):
    norm = pd.DataFrame(index=table.index, columns=table.columns, dtype=float)
    for protein in table.columns:
        col = table[protein].dropna()
        if len(col) < 2:
            continue
        denom = col.max() - col.min()
        if denom == 0 or not np.isfinite(denom):
            continue
        norm[protein] = (col - col.min()) / denom
    return norm


def run(ens_base=None, outdir=None, exp_dir=None, rg_json=None, score_file="scores_trials.csv",
        exclude_methods=None, proteins=None):
    _t = os.path.join(os.path.dirname(__file__), '..', '..')
    ens_base = ens_base or os.path.join(_t, 'output', 'ensembles')
    outdir = outdir or os.path.join(_t, 'output')
    exp_dir = exp_dir or os.environ.get("IDPFORGE_EXP_DATA", os.path.join(_t, 'Data', 'exp'))
    canon = derive_canonical_sets(exp_dir)
    if proteins:  # restrict every category to a fixed protein subset (e.g. Table 2 MD subset)
        keep = set(proteins)
        canon = {cat: [p for p in plist if p in keep] for cat, plist in canon.items()}

    method_paths = discover_methods(ens_base)
    if exclude_methods:  # e.g. drop the a99SB-disp MD reference ('desres') from Table 1
        drop = set(exclude_methods)
        method_paths = {m: p for m, p in method_paths.items() if m not in drop}
    all_scores = {}
    for name, path in method_paths.items():
        scores = load_scores(path, score_file)
        if scores:
            all_scores[name] = scores
            print(f"Loaded {name}: {len(scores)} proteins")
    if not all_scores:
        print("No scores found!")
        return

    methods = sorted(all_scores.keys())
    canonical_all = sorted(set().union(*[set(canon[c]) for c in ["CS", "JC", "NOE/PRE", "FRET"]]))
    print(f"\nMethods: {len(methods)}, canonical proteins (union): {len(canonical_all)}")
    for cat in ["CS", "JC", "NOE", "PRE", "NOE/PRE", "FRET"]:
        print(f"  {cat:<10s} {len(canon.get(cat, []))} proteins  {canon.get(cat, [])}")

    # raw per-category tables (method x protein)
    raw = {cat: pd.DataFrame(index=methods, columns=canon[cat], dtype=float)
           for cat in ["CS", "JC", "NOE/PRE", "FRET"]}
    for m in methods:
        for protein, df in all_scores[m].items():
            if protein in canon["CS"] and "cs_score" in df.columns:
                raw["CS"].loc[m, protein] = df["cs_score"].mean()
            if protein in canon["JC"] and "jc_score" in df.columns:
                raw["JC"].loc[m, protein] = df["jc_score"].mean()
            if protein in canon["NOE/PRE"]:
                np_sum, present = 0.0, False
                if protein in canon["NOE"] and "noe_score" in df.columns:
                    np_sum += df["noe_score"].mean(); present = True
                if protein in canon["PRE"] and "pre_score" in df.columns:
                    np_sum += df["pre_score"].mean(); present = True
                if present:
                    raw["NOE/PRE"].loc[m, protein] = np_sum
            if protein in canon["FRET"] and "fret_score" in df.columns:
                raw["FRET"].loc[m, protein] = df["fret_score"].mean()

    norm = {cat: normalize_per_protein(tbl) for cat, tbl in raw.items()}

    headline = ["CS", "JC", "NOE/PRE"]
    summary = pd.DataFrame(index=methods, columns=["Total"] + headline + ["FRET"], dtype=float)
    counts = pd.DataFrame(index=methods, columns=headline + ["FRET"], dtype=int)
    for cat in headline + ["FRET"]:
        summary[cat] = norm[cat].mean(axis=1)
        counts[cat] = norm[cat].notna().sum(axis=1)

    # per-protein TOTAL: aggregate all observables, then normalize
    total_proteins = canon["CS"]
    raw_total = pd.DataFrame(index=methods, columns=total_proteins, dtype=float)
    for m in methods:
        for protein in total_proteins:
            df = all_scores[m].get(protein)
            if df is None:
                continue
            agg, present = 0.0, False
            for col in ["cs_score", "jc_score", "noe_score", "pre_score", "fret_score"]:
                if col in df.columns:
                    agg += df[col].mean(); present = True
            if present:
                raw_total.loc[m, protein] = agg
    norm_total = normalize_per_protein(raw_total)
    summary["Total"] = norm_total.mean(axis=1)

    print("\n=== Proteins contributing to each category (per method) ===")
    print(counts.to_string())

    # chi2_CS (Eqn 12): mean over trials of the stored per-shift cs_chi2
    #   cs_chi2 = (1/N) Σ_all_shifts (δ_exp - δ_bc)^2 / σ_bc,atom^2
    chi2_table = pd.DataFrame(index=methods, columns=canon["CS"], dtype=float)
    for m in methods:
        for protein in canon["CS"]:
            df = all_scores[m].get(protein)
            if df is not None and "cs_chi2" in df.columns:
                chi2_table.loc[m, protein] = df["cs_chi2"].mean()
    summary["chi2_CS"] = chi2_table.mean(axis=1)

    if rg_json and os.path.exists(rg_json):
        with open(rg_json) as f:
            exp_rg = {p: (v[0], v[1]) for p, v in json.load(f)["exp_rg"].items()}
    else:
        exp_rg = {}
        if rg_json:
            print(f"WARNING: {rg_json} not found — Rg column will be empty")
    # Rg column: mean Rg from each method's rg_trials.csv, over proteins with an
    # exp Rg target. Flat-bottom %|dRg|/Rg.
    rg_proteins = sorted(set(exp_rg))
    if proteins:
        rg_proteins = [p for p in rg_proteins if p in set(proteins)]
    rg_table = pd.DataFrame(index=methods, columns=rg_proteins, dtype=float)
    for m in methods:
        for protein in rg_proteins:
            rg_mean = load_rg_trials(method_paths[m], protein)
            if rg_mean is None or not np.isfinite(rg_mean):
                continue
            rg_e, sigma_rg = exp_rg[protein]
            dev = max(abs(rg_mean - rg_e) - sigma_rg, 0.0)  # flat-bottom (Eqn. 5)
            rg_table.loc[m, protein] = dev / rg_e * 100.0
    summary["pct_dRg"] = rg_table.mean(axis=1)

    cols = ["Total", "CS", "JC", "NOE/PRE", "FRET", "chi2_CS", "pct_dRg"]
    print("\n" + "=" * 90 + "\nPAPER-STYLE BENCHMARK TABLE\n" + "=" * 90)
    print(f"{'Method':<32s} " + "  ".join(f"{c:>9s}" for c in cols))
    print("-" * 90)
    for m, row in summary.sort_values("Total", ascending=False).iterrows():
        cells = [f"{row[c]:>9.3f}" if np.isfinite(row[c]) else f"{'-':>9s}" for c in cols]
        print(f"{m:<32s} " + "  ".join(cells))
    print("\nNormalized X-EISD: higher = better; chi2_CS, pct_dRg: lower = better")
    print("Total = mean(CS, JC, NOE/PRE); FRET shown separately")

    os.makedirs(outdir, exist_ok=True)
    summary.to_csv(os.path.join(outdir, "xeisd_summary.csv"))
    for cat, tbl in raw.items():
        safe = cat.replace("/", "_")
        tbl.to_csv(os.path.join(outdir, f"xeisd_raw_{safe}.csv"))
        norm[cat].to_csv(os.path.join(outdir, f"xeisd_normalized_{safe}.csv"))
    chi2_table.to_csv(os.path.join(outdir, "xeisd_chi2_CS.csv"))
    rg_table.to_csv(os.path.join(outdir, "rg_pct_deviation.csv"))
    raw_total.to_csv(os.path.join(outdir, "xeisd_total_raw.csv"))
    norm_total.to_csv(os.path.join(outdir, "xeisd_total_heatmap.csv"))
    print(f"\nSaved benchmark tables to {outdir}/")
