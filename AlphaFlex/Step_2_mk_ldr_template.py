"""
Step 2: Template Generation Dispatcher
======================================

Reads the labeled database and generates .npz structure templates for each 
disordered region. These templates are required for the diffusion model (Step 3).

Workflow:
  1. Loads a subset of IDs (from --start-index or split files).
  2. Maps IDs to AlphaFold PDB files.
  3. Dispatches subprocess calls to 'mk_ldr_template.py' (for Tails/Loops)
     or 'mk_flex_template.py' (for Linkers).
  4. Aggregates fully disordered proteins (Category 0) into a separate JSON list.
"""

import json
import os
import sys
import subprocess
import argparse
import re
import mdtraj as md
from glob import glob

# --- Load Centralized Configuration ---
try:
    import config as cfg
except ImportError:
    print("CRITICAL ERROR: 'project_config.py' not found.")
    sys.exit(1)

# =============================================================================
# LOGGING HELPER
# =============================================================================
def log(msg, force=False):
    """
    Prints message only if VERBOSE is True, or if 'force' is True.
    """
    if getattr(cfg, 'VERBOSE', True) or force:
        print(msg)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_pdb_map(pdb_dir):
    log(f"Scanning PDB library: {pdb_dir}", force=True)
    mapping = {}
    pattern_af = re.compile(r"AF-([A-Z0-9]+)-F1")
    pattern_simple = re.compile(r"([A-Z0-9]+)\.pdb")
    
    files = glob(os.path.join(pdb_dir, "*.pdb"))
    for f in files:
        base = os.path.basename(f)
        m = pattern_af.search(base)
        if m:
            mapping[m.group(1)] = f
        else:
            m2 = pattern_simple.match(base)
            if m2:
                mapping[m2.group(1)] = f
                
    log(f"  -> Mapped {len(mapping)} PDB files.", force=True)
    return mapping

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Step 2: IDPForge Template Generation")
    parser.add_argument("--start-index", type=int, default=None, 
                        help="Force start at specific index (1-based).")
    args = parser.parse_args()

    # 1. Validation
    if not os.path.exists(cfg.LABELED_DB_PATH):
        print(f"ERROR: Labeled DB not found at {cfg.LABELED_DB_PATH}")
        sys.exit(1)

    if not os.path.exists(cfg.ID_LISTS_DIR):
        print(f"ERROR: ID List directory not found: {cfg.ID_LISTS_DIR}")
        sys.exit(1)

    # 2. Load Resources
    log(f"Loading database...", force=True)
    with open(cfg.LABELED_DB_PATH, 'r') as f:
        labeled_db = json.load(f)

    log(f"Loading work queue from: {cfg.ID_LISTS_DIR}", force=True)
    all_txt_files = glob(os.path.join(cfg.ID_LISTS_DIR, "*.txt"))
    
    if not all_txt_files:
        print("[!] No .txt files found.")
        sys.exit(1)

    all_ids_set = set()
    for txt in all_txt_files:
        with open(txt, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            all_ids_set.update(lines)
            log(f"  - Loaded {len(lines)} IDs from {os.path.basename(txt)}")
    
    all_prot_ids = sorted(list(all_ids_set))
    log(f"  -> Total Unique IDs to process: {len(all_prot_ids)}", force=True)

    # Build PDB Map
    id_to_pdb = get_pdb_map(cfg.PDB_LIBRARY_PATH)

    # 3. Setup Output & Resume Logic
    os.makedirs(cfg.TEMPLATE_OUTPUT_DIR, exist_ok=True)
    progress_file = os.path.join(cfg.TEMPLATE_OUTPUT_DIR, "Step_2_progress.txt")
    
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
    if os.path.exists(cfg.IDP_CASES_LIST_PATH):
        try:
            with open(cfg.IDP_CASES_LIST_PATH, 'r') as f:
                idp_cases = json.load(f)
        except: pass

    # 5. Processing Loop
    stats = {'processed': 0, 'created': 0, 'skipped_cache': 0, 
             'skipped_error': 0, 'skipped_timeout': 0, 'idp_found': 0}

    print(f"\n--- Starting Batch ({len(ids_to_process)} proteins) ---")

    for i, prot_id in enumerate(ids_to_process):
        current_global_idx = start_idx + i
        
        # Save Progress
        with open(progress_file, 'w') as pf:
            pf.write(prot_id)

        # Silent Mode: Print progress update every 100 items
        if not getattr(cfg, 'VERBOSE', True):
            if i % 100 == 0:
                print(f"  ... Processed {i}/{len(ids_to_process)} proteins ...")

        # Verbose Mode: Print every item
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

        for idr in labeled_idrs:
            idr_type = idr.get('type')
            rng = idr.get('range')
            if not idr_type or not rng: continue

            subdir_name = idr_type.replace(" ", "_") 
            out_dir = os.path.join(cfg.TEMPLATE_OUTPUT_DIR, subdir_name)
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
                timeout = cfg.TIMEOUT_STATIC_TEMPLATE
            elif idr_type == "Linker IDR":
                script = cfg.SCRIPT_FLEX_TEMPLATE
                timeout = cfg.TIMEOUT_DYNAMIC_TEMPLATE
            else: continue

            cmd = [
                cfg.PYTHON_EXEC, script,
                pdb_path,
                f"{rng[0]}-{rng[1]}",
                out_path,
                "--nconf", str(cfg.TEMPLATE_N_CONFS)
            ]

            try:
                log(f"  -> Generating {idr_type} {rng}...", force=False)
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                
                if res.returncode == 0:
                    log("     Done.")
                    stats['created'] += 1
                else:
                    # ALWAYS print errors, even in silent mode
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
        
        stats['processed'] += 1
        
        if i % 10 == 0:
            with open(cfg.IDP_CASES_LIST_PATH, 'w') as f:
                json.dump(idp_cases, f, indent=4)

    # 6. Finalization
    print(f"\n--- Step 2 Batch Complete ---")
    print(f" Processed Proteins: {stats['processed']}")
    print(f" Templates Created:  {stats['created']}")
    print(f" IDP Cases Added:    {stats['idp_found']}")
    print(f" Skipped (Exists):   {stats['skipped_cache']}")
    print(f" Skipped (Error):    {stats['skipped_error']}")
    print(f" Skipped (Timeout):  {stats['skipped_timeout']}")

    with open(cfg.IDP_CASES_LIST_PATH, 'w') as f:
        json.dump(idp_cases, f, indent=4)

    if start_idx + len(ids_to_process) >= len(all_prot_ids):
        if os.path.exists(progress_file): os.remove(progress_file)

if __name__ == "__main__":
    main()