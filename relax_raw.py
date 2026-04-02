#!/usr/bin/env python3
"""
Relax existing raw PDB files using the same Amber relaxation as sample_ldr.py.
Optionally validates structures with the same repair+validate pipeline as output_to_pdb.

Usage:
    python relax_raw.py ./A2RUH7_raw path/to/template.npz configs/sample.yml --cuda
    python relax_raw.py ./A2RUH7_raw path/to/template.npz configs/sample.yml --cuda --validate --verbose
    python relax_raw.py ./A2RUH7_raw path/to/template.npz configs/sample.yml --cuda --output_dir ./A2RUH7_relaxed
"""

import os
import sys
import argparse
import yaml
import re
import time
import numpy as np
import torch
import ml_collections as mlc

from glob import glob
from openfold.np.protein import from_pdb_string
from idpforge.utils.relax import relax_protein

sys.stdout.reconfigure(line_buffering=True)

_HIS_RESNAMES = {'HIS', 'HID', 'HIE', 'HIP'}


def main():
    parser = argparse.ArgumentParser(
        description="Relax existing *_raw.pdb files with Amber minimization.")
    parser.add_argument("raw_dir", help="Directory containing *_raw.pdb files")
    parser.add_argument("template_npz", help="Template .npz file (for mask → exclude_residues)")
    parser.add_argument("sample_cfg", help="Config YAML (e.g. configs/sample.yml)")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for relaxation")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: same as raw_dir)")
    parser.add_argument("--validate", action="store_true",
                        help="Run structural repair + validation after relaxation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed validation output")
    args = parser.parse_args()

    # 1. Load template mask → exclude_residues
    fold_data = np.load(args.template_npz, allow_pickle=True)
    mask = fold_data["mask"]  # True = folded, False = IDR
    exclude_residues = np.where(~mask)[0].tolist()
    print(f"[relax] Loaded template: {args.template_npz}")
    print(f"[relax] exclude_residues: {len(exclude_residues)} IDR residues (free to move)")

    # 2. Load relax config from YAML
    with open(args.sample_cfg, "r") as f:
        settings = yaml.safe_load(f)

    relax_config = settings["relax"]
    relax_config["exclude_residues"] = exclude_residues
    relax_opts = mlc.ConfigDict(relax_config)
    print(f"[relax] Relax config: stiffness={relax_config.get('stiffness')}, "
          f"max_outer_iterations={relax_config.get('max_outer_iterations')}")

    # 3. Find raw PDBs, sorted numerically
    raw_files = sorted(
        glob(os.path.join(args.raw_dir, "*_raw.pdb")),
        key=lambda f: int(re.search(r"/(\d+)_raw\.pdb$", f).group(1))
    )
    if not raw_files:
        print(f"[relax] No *_raw.pdb files found in {args.raw_dir}")
        sys.exit(1)

    print(f"[relax] Found {len(raw_files)} raw PDB files")

    # 4. Setup output
    output_dir = args.output_dir or args.raw_dir
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # 5. Relax (and optionally validate) each raw PDB
    success = 0
    failed = 0
    validated = 0
    validate_failed = 0

    for raw_path in raw_files:
        basename = os.path.basename(raw_path)
        stem = basename.replace("_raw.pdb", "")
        relaxed_path = os.path.join(output_dir, f"{stem}_relaxed.pdb")

        if args.verbose:
            print("-" * 60, flush=True)
            print(f"     [Conformer {stem}]", flush=True)

        print(f"\n[relax] Processing {basename}...")

        with open(raw_path, "r") as f:
            pdb_str = f.read()

        unrelaxed_protein = from_pdb_string(pdb_str)
        viol_mask = ~mask

        result = relax_protein(
            relax_opts, device, unrelaxed_protein,
            output_dir, stem, viol_mask=viol_mask,
        )

        if result != 1:
            print(f"[relax]   → FAILED (violation threshold or minimization error)")
            failed += 1
            continue

        print(f"[relax]   → {stem}_relaxed.pdb written")
        success += 1

        # --- Optional validation pipeline ---
        if not args.validate:
            continue

        from openmm.app import PDBFile as OMMPDBFile
        from idpforge.utils.structure_validation import (
            validate_structure_post_relax, check_bond_integrity,
        )
        from idpforge.utils.structure_repair import (
            repair_chirality, fix_histidine_naming,
        )

        needs_rerelax = False

        # 1a. Bond integrity -> find broken HIS residues
        try:
            chk = OMMPDBFile(relaxed_path)
            broken = check_bond_integrity(chk.topology, chk.positions)
            his_resids = {b['resid'] for b in broken
                          if b['resname'] in _HIS_RESNAMES
                          or b.get('resname2', '') in _HIS_RESNAMES}
        except Exception:
            his_resids = set()

        # 1b. Chirality repair
        n_chiral = repair_chirality(relaxed_path)
        if n_chiral > 0:
            if args.verbose:
                print(f"       [REPAIR] Flipped {n_chiral} D-isomer(s)", flush=True)
            needs_rerelax = True

        # 1c. HIS naming fix
        if his_resids:
            n_his = fix_histidine_naming(relaxed_path, his_resids)
            if n_his > 0:
                needs_rerelax = True

        # Re-relax if repairs applied
        if needs_rerelax:
            if args.verbose:
                print(f"       [RE-RELAX] Re-relaxing after repairs...", flush=True)
            repaired_prot = from_pdb_string(open(relaxed_path).read())
            os.remove(relaxed_path)
            relax_protein(
                relax_opts, device, repaired_prot,
                output_dir, stem, viol_mask=viol_mask,
            )
            if not os.path.isfile(relaxed_path):
                if args.verbose:
                    print(f"       [RE-RELAX] Failed, discarding.", flush=True)
                validate_failed += 1
                continue
        elif args.verbose:
            print(f"       [PRE-VALIDATION] All checks clean.", flush=True)

        # Structural validation
        t0 = time.perf_counter()
        try:
            chk = OMMPDBFile(relaxed_path)
            is_valid, info = validate_structure_post_relax(
                chk.topology, chk.positions, pdb_path=relaxed_path,
                full_report=True
            )
        except Exception as e:
            is_valid = False
            info = {"reason": str(e), "chirality_pass": False,
                    "bonds_pass": False, "clash_pass": False,
                    "knot_pass": False}
        elapsed = time.perf_counter() - t0

        if args.verbose:
            chiral_pass = info.get("chirality_pass", True)
            bonds_pass = info.get("bonds_pass", True)
            clash_pass = info.get("clash_pass", True)
            knot_pass = info.get("knot_pass", True)
            clash_count = info.get("num_clashes", "?")
            clash_score = info.get("clash_score", 0.0)
            n_broken = info.get("num_broken_bonds", 0)
            knot_type = info.get("knot_type", "None")
            threshold = 10.0

            chiral_str = "PASS" if chiral_pass else "FAIL (D-Amino detected)"
            bonds_str = "PASS" if bonds_pass else f"FAIL ({n_broken} broken)"
            clash_str = "PASS" if clash_pass else "FAIL"
            knot_str = "PASS" if knot_pass else f"FAIL ({knot_type})"

            print(f"       [TIMING] Validation: {elapsed:.2f}s", flush=True)
            print(f"       [POST-MIN CHECK] Validation Results", flush=True)
            print(f"         - Chirality: {chiral_str}", flush=True)
            print(f"         - Bonds:     {bonds_str}", flush=True)
            print(f"         - Clashes:   {clash_count}  (Score: {clash_score:.2f} | Limit: {threshold:.1f})", flush=True)
            print(f"         - Topology:  {knot_str}", flush=True)

        if is_valid:
            validated += 1
            validated_path = relaxed_path.replace("_relaxed.pdb", "_validated.pdb")
            os.rename(relaxed_path, validated_path)
            if args.verbose:
                print(f"       [RESULT] SUCCESS!", flush=True)
        else:
            reason = info.get("reason", "Unknown")
            if args.verbose:
                print(f"       [RESULT] FAILED ({reason}) [Thresh: 10.00]", flush=True)
            os.remove(relaxed_path)
            validate_failed += 1

    # Summary
    if args.validate:
        print(f"\n[relax] Done. {success} relaxed, {failed} relax-failed, "
              f"{validated} validated, {validate_failed} validation-failed "
              f"out of {len(raw_files)} total.")
    else:
        print(f"\n[relax] Done. {success} relaxed, {failed} failed out of {len(raw_files)} total.")


if __name__ == "__main__":
    main()
