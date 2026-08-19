"""Step 2: generate per-IDR .npz structure templates for the diffusion model (Step 3)."""

import json
import os
import sys
import subprocess
import argparse
import time
import re
import tempfile
import mdtraj as md
from glob import glob

try:
    import config as cfg
except ImportError:
    print("CRITICAL ERROR: 'project_config.py' not found.")
    sys.exit(1)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def log(msg, force=False):
    if getattr(cfg, 'VERBOSE', True) or force:
        print(msg)

def get_pdb_map(pdb_dir):
    log(f"Scanning PDB library: {pdb_dir}", force=True)
    mapping = {}
    pattern_af = re.compile(r"AF-([A-Z0-9]+)-F1")
    pattern_simple = re.compile(r"([A-Z0-9]+)\.pdb")
    
    files = glob(os.path.join(pdb_dir, "*.pdb"))
    for f in files:
        base = os.path.basename(f)
        m = pattern_af.search(base)
        if m: mapping[m.group(1)] = f
        else:
            m2 = pattern_simple.match(base)
            if m2: mapping[m2.group(1)] = f
                
    log(f"  -> Mapped {len(mapping)} PDB files.", force=True)
    return mapping

def main(args):
    IS_VERBOSE = args.verbose
    if IS_VERBOSE:
        print(f"   [STATUS] Verbose Mode: ON (Detailed Logs Enabled)", flush=True)

    # --- Truncation setup ---
    truncate = bool(getattr(args, "truncate", False))
    trunc_tmp_dir = None
    _extract_ca_by_resid = _compute_window = _slice_pdb_file = None
    if truncate:
        try:
            if _REPO_ROOT not in sys.path:
                sys.path.insert(0, _REPO_ROOT)
            from Analysis.screen_knots import extract_ca_by_resid as _extract_ca_by_resid
            from utils.truncate import compute_idr_truncation_window as _compute_window
            from slice_pdb import slice_pdb_file as _slice_pdb_file
            trunc_tmp_dir = os.path.join(args.output_dir, "_trunc_tmp")
            os.makedirs(trunc_tmp_dir, exist_ok=True)
            print("   [STATUS] Truncation: ON (IDR + adjacent folded domains; .trunc.json sidecars)", flush=True)
        except Exception as e:
            print(f"   [!] Truncation requested but unavailable ({e}); using full-length templates.", flush=True)
            truncate = False

    # 1. Validation
    if not os.path.exists(args.labeled_db):
        print(f"ERROR: Labeled DB not found at {args.labeled_db}")
        sys.exit(1)

    if args.id_file:
        if not os.path.exists(args.id_file):
            print(f"ERROR: ID file not found: {args.id_file}")
            sys.exit(1)
    elif not args.id_lists_dir or not os.path.exists(args.id_lists_dir):
        print(f"ERROR: no input list -- set config RUN_ID_FILE / --id_file, or pass --id_lists_dir.")
        sys.exit(1)

    # 2. Load Resources
    log(f"Loading database...", force=True)
    with open(args.labeled_db, 'r') as f:
        labeled_db = json.load(f)

    if args.id_file:
        log(f"Loading work queue from file: {args.id_file}", force=True)
        all_txt_files = [args.id_file]
    else:
        log(f"Loading work queue from: {args.id_lists_dir}", force=True)
        all_txt_files = glob(os.path.join(args.id_lists_dir, "*.txt"))

    if not all_txt_files:
        print("[!] No .txt files found.")
        sys.exit(1)

    all_ids_set = set()
    for txt in all_txt_files:
        with open(txt, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            all_ids_set.update(lines)
            if IS_VERBOSE:
                log(f"  - Loaded {len(lines)} IDs from {os.path.basename(txt)}")

    all_prot_ids = sorted(list(all_ids_set))
    log(f"  -> Total Unique IDs to process: {len(all_prot_ids)}", force=True)

    id_to_pdb = get_pdb_map(args.pdb_library)

    # 3. Setup Output & Resume
    os.makedirs(args.output_dir, exist_ok=True)
    progress_file = os.path.join(args.output_dir, "Step_2_progress.txt")
    
    start_idx = 0
    if args.start_index is not None:
        start_idx = args.start_index - 1
        print(f"Manual override: Starting at index {args.start_index}")
    elif os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as pf:
                last_id = pf.read().strip()
                if last_id in all_prot_ids:
                    start_idx = all_prot_ids.index(last_id)
                    print(f"Resuming from ID: {last_id} (Index {start_idx + 1})")
        except: pass

    ids_to_process = all_prot_ids[start_idx:]
    if not ids_to_process:
        print("Nothing to process!")
        sys.exit(0)

    # 4. Load Cache
    idp_cases = {}
    idp_cases_path = os.path.join(args.output_dir, os.path.basename(cfg.IDP_CASES_LIST_PATH))
    if os.path.exists(idp_cases_path):
        try:
            with open(idp_cases_path, 'r') as f: idp_cases = json.load(f)
        except: pass

    stats = {'processed': 0, 'created': 0, 'skipped_cache': 0, 
             'skipped_error': 0, 'skipped_timeout': 0, 'idp_found': 0}

    print(f"\n--- Starting Batch ({len(ids_to_process)} proteins) ---")

    for i, prot_id in enumerate(ids_to_process):
        current_global_idx = start_idx + i
        
        with open(progress_file, 'w') as pf: pf.write(prot_id)

        log(f"\n[{current_global_idx + 1}/{len(all_prot_ids)}] Processing {prot_id}...")

        if prot_id not in labeled_db:
            log(f"  [!] Skipped: Not in labeled DB.")
            stats['skipped_error'] += 1
            continue

        data = labeled_db[prot_id]
        category = data.get('category')
        labeled_idrs = data.get('labeled_idrs', [])
        
        # --- CASE 0: Full IDP ---
        if category == 0:
            sequence = data.get('sequence')
            if not sequence and prot_id in id_to_pdb:
                 try:
                    t = md.load(id_to_pdb[prot_id])
                    sequence = t.top.to_fasta(chain=0)
                 except: pass

            if not sequence:
                log("  [!] Skipped IDP: No sequence.")
                stats['skipped_error'] += 1
                continue
            
            if prot_id not in idp_cases:
                idp_cases[prot_id] = sequence
                stats['idp_found'] += 1
                log("  -> Category 0: Added to list.")
            else:
                log("  -> Category 0: Already in list.")
                stats['skipped_cache'] += 1
            continue

        # --- CASE 1, 2, 3 ---
        pdb_path = id_to_pdb.get(prot_id)
        if not pdb_path:
            log(f"  [!] Skipped Cat {category}: No PDB found.")
            stats['skipped_error'] += 1
            continue

        # Observed CA residues
        all_res = None
        if truncate:
            try:
                all_res = set(_extract_ca_by_resid(pdb_path).keys())
            except Exception as e:
                log(f"  [!] {prot_id}: could not read CA resids ({e}); truncation off for this protein.")
                all_res = None

        for idr in labeled_idrs:
            idr_type = idr.get('type')
            rng = idr.get('range')
            if not idr_type or not rng: continue

            subdir_name = idr_type.replace(" ", "_") 
            out_dir = os.path.join(args.output_dir, subdir_name)
            os.makedirs(out_dir, exist_ok=True)
            
            fname = f"{prot_id}_idr_{rng[0]}-{rng[1]}.npz"
            out_path = os.path.join(out_dir, fname)

            if os.path.exists(out_path):
                log(f"  -> Exists: {subdir_name}/{fname}")
                stats['skipped_cache'] += 1
                continue

            # Select Script
            if idr_type in ["Tail IDR", "Loop IDR"]:
                script = cfg.SCRIPT_STATIC_TEMPLATE
                timeout = args.timeout_static
                script_name = "mk_ldr (Static)"
            elif idr_type == "Linker IDR":
                script = cfg.SCRIPT_FLEX_TEMPLATE
                timeout = args.timeout_dynamic
                script_name = "mk_flex (Dynamic)"
            else: continue

            # --- Truncation ---
            template_pdb = pdb_path
            disorder_range = f"{rng[0]}-{rng[1]}"
            sidecar = None
            tmp_pdb = None
            if truncate and not args.max_residues and all_res:
                try:
                    win = _compute_window(labeled_idrs, idr, all_res)
                    if not win["is_noop"]:
                        fd, tmp_pdb = tempfile.mkstemp(
                            suffix=".pdb",
                            prefix=f"{prot_id}_{win['trunc_lo']}-{win['trunc_hi']}_",
                            dir=trunc_tmp_dir)
                        os.close(fd)
                        _slice_pdb_file(pdb_path, win["trunc_lo"], win["trunc_hi"],
                                        tmp_pdb, renumber=False)
                        template_pdb = tmp_pdb
                        slo, shi = win["idr_range_seq"]
                        disorder_range = f"{slo}-{shi}"
                        sidecar = {
                            "offset": win["offset"],
                            "trunc_lo": win["trunc_lo"], "trunc_hi": win["trunc_hi"],
                            "idr_range_full": list(win["idr_range_full"]),
                            "idr_range_seq": list(win["idr_range_seq"]),
                            "retained_labels": win["retained_labels"],
                        }
                        if IS_VERBOSE:
                            keep = ",".join(win["retained_labels"]) or "-"
                            print(f"  [TRUNC] {prot_id} idr {rng[0]}-{rng[1]} -> window "
                                  f"{win['trunc_lo']}-{win['trunc_hi']} (offset {win['offset']}, "
                                  f"seq {slo}-{shi}, keep {keep})", flush=True)
                    elif IS_VERBOSE:
                        print(f"  [TRUNC] {prot_id} idr {rng[0]}-{rng[1]}: window spans whole "
                              f"protein -> no truncation.", flush=True)
                except Exception as e:
                    print(f"  [!] {prot_id} idr {rng[0]}-{rng[1]}: truncation failed ({e}); "
                          f"using full-length PDB.", flush=True)
                    template_pdb, disorder_range, sidecar, tmp_pdb = pdb_path, f"{rng[0]}-{rng[1]}", None, None

            cmd = [
                cfg.PYTHON_EXEC, script,
                template_pdb,
                disorder_range,
                out_path,
                "--nconf", str(args.n_confs)
            ]
            if script == cfg.SCRIPT_STATIC_TEMPLATE:
                cmd += ["--seed_skew", str(args.seed_skew)]
            if args.max_residues:
                cmd += ["--max_residues", str(args.max_residues)]
            if args.fold_per_side:
                cmd += ["--fold_per_side", str(args.fold_per_side)]

            try:
                if IS_VERBOSE:
                    print(f"  [INFO] Dispatching Template | Script: {script_name} | Timeout: {timeout}s", flush=True)

                t_gen_start = time.time()
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                t_gen_end = time.time()

                if res.returncode == 0:
                    if sidecar is not None:
                        with open(out_path[:-4] + ".trunc.json", "w") as sf:
                            json.dump(sidecar, sf, indent=2)
                    if IS_VERBOSE:
                        print(f"  [TIMING] Template Generation: {t_gen_end - t_gen_start:.2f}s", flush=True)
                    else:
                        log("     Done.")
                    stats['created'] += 1
                else:
                    print(f"[!] FAILED: {prot_id} {idr_type}")
                    err_msg = res.stderr.strip()[:200] if res.stderr else "Unknown Error"
                    print(f"    Error: {err_msg}...")
                    stats['skipped_error'] += 1

            except subprocess.TimeoutExpired:
                print(f"[!] TIMEOUT: {prot_id} {idr_type} (> {timeout}s)")
                stats['skipped_timeout'] += 1
            except Exception as e:
                print(f"[!] EXCEPTION: {prot_id} - {e}")
                stats['skipped_error'] += 1
            finally:
                if tmp_pdb and os.path.exists(tmp_pdb):
                    try:
                        os.remove(tmp_pdb)
                    except OSError:
                        pass
        
        stats['processed'] += 1
        
        if i % 10 == 0:
            with open(idp_cases_path, 'w') as f:
                json.dump(idp_cases, f, indent=4)

    # 6. Finalization
    print(f"\n--- Step 2 Batch Complete ---")
    print(f" Processed Proteins: {stats['processed']}")
    print(f" Templates Created:  {stats['created']}")
    print(f" IDP Cases Added:    {stats['idp_found']}")
    print(f" Skipped (Exists):   {stats['skipped_cache']}")
    print(f" Skipped (Error):    {stats['skipped_error']}")
    print(f" Skipped (Timeout):  {stats['skipped_timeout']}")

    with open(idp_cases_path, 'w') as f:
        json.dump(idp_cases, f, indent=4)

    if start_idx + len(ids_to_process) >= len(all_prot_ids):
        if os.path.exists(progress_file): os.remove(progress_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: IDPForge Template Generation")
    parser.add_argument("--start-index", type=int, default=None,
                        help="Force start at specific index (1-based).")
    parser.add_argument("--labeled_db", default=cfg.LABELED_DB_PATH,
                        help=f"Path to labeled database JSON (default: {cfg.LABELED_DB_PATH}).")
    parser.add_argument("--id_file", default=cfg.RUN_ID_FILE,
                        help=f"Single newline-separated ID list to process (default from config: "
                             f"{cfg.RUN_ID_FILE}). Pass '' with --id_lists_dir to union a directory instead.")
    parser.add_argument("--id_lists_dir", default=None,
                        help="Optional: directory of ID list .txt files to union (used only when "
                             "--id_file is empty).")
    parser.add_argument("--pdb_library", default=cfg.PDB_LIBRARY_PATH,
                        help=f"Path to AlphaFold PDB library (default: {cfg.PDB_LIBRARY_PATH}).")
    parser.add_argument("--output_dir", default=cfg.TEMPLATE_OUTPUT_DIR,
                        help=f"Output directory for templates (default: {cfg.TEMPLATE_OUTPUT_DIR}).")
    parser.add_argument("--n_confs", type=int, default=cfg.TEMPLATE_N_CONFS,
                        help=f"Number of conformations per template (default: {cfg.TEMPLATE_N_CONFS}).")
    parser.add_argument("--seed_skew", type=float, default=cfg.TEMPLATE_SEED_SKEW,
                        help=f"Tail/loop seed-distance skew on [d/2, d] -> mk_ldr (0=all d/2, 1=all d, "
                             f"0.5=uniform; default {cfg.TEMPLATE_SEED_SKEW}). Linkers (mk_flex) ignore it.")
    parser.add_argument("--timeout_static", type=int, default=cfg.TIMEOUT_STATIC_TEMPLATE,
                        help=f"Timeout for static templates in seconds (default: {cfg.TIMEOUT_STATIC_TEMPLATE}).")
    parser.add_argument("--timeout_dynamic", type=int, default=cfg.TIMEOUT_DYNAMIC_TEMPLATE,
                        help=f"Timeout for dynamic templates in seconds (default: {cfg.TIMEOUT_DYNAMIC_TEMPLATE}).")
    parser.add_argument("--verbose", action="store_true", default=cfg.VERBOSE,
                        help="Enable detailed per-protein logging.")
    parser.add_argument("--truncate", dest="truncate", action="store_true", default=None,
                        help="Build each per-IDR template on the IDR + its adjacent folded "
                             "domain(s) only (shrinks the Step-3 relax; stitched back in Step 4). "
                             "Default: cfg.TRUNCATE_TO_ADJACENT.")
    parser.add_argument("--no_truncate", dest="truncate", action="store_false",
                        help="Disable truncation; build templates on the full-length PDB.")
    parser.add_argument("--max_residues", type=int, default=cfg.TEMPLATE_MAX_RESIDUES,
                        help="Graft-mode truncation (recommended for the graft stitcher): pass this "
                             "cap to mk_ldr / mk_flex so each template keeps the IDR + a "
                             "junction-adjacent fold block (>=50 per junction) sized to the cap and "
                             "records the graft spec in the .npz. Step 3 (sample_ldr) then emits "
                             "_truncation.json and Step 4 reconstruct_truncated_pool grafts + "
                             "validates. Bypasses the old window pre-slice / .trunc.json path.")
    parser.add_argument("--fold_per_side", type=int, default=cfg.TEMPLATE_FOLD_PER_SIDE,
                        help="mk_ldr only: keep EXACTLY this many fold residues per junction "
                             "('IDR + N fold' mode) instead of filling the --max_residues budget. "
                             f"Smaller/more in-distribution. Default from config: {cfg.TEMPLATE_FOLD_PER_SIDE}.")
    args = parser.parse_args()
    if args.truncate is None:
        args.truncate = bool(getattr(cfg, "TRUNCATE_TO_ADJACENT", False))
    main(args)