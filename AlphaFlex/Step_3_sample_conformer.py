"""
Step 3: Diffusion Sampling (Dispatcher)
=======================================

Recursively scans for templates created in Step 2 and dispatches 
'sample_ldr.py' to generate conformers for each one.

"""

import os
import sys
import subprocess
import argparse
import re
from glob import glob

# --- Load Centralized Configuration ---
try:
    import config as cfg
except ImportError:
    print("CRITICAL ERROR: 'config.py' not found.")
    sys.exit(1)

# =============================================================================
# LOGGING HELPER
# =============================================================================
def log(msg, force=False):
    """Prints message only if VERBOSE is True, or if 'force' is True."""
    if getattr(cfg, 'VERBOSE', True) or force:
        print(msg)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Step 3: IDPForge Sampling Dispatcher")
    parser.add_argument("--input-dir", type=str, default=None, 
                        help="Optional: Override input directory.")
    args = parser.parse_args()

    # 1. Setup
    log(f"--- Step 3: Sampling Dispatcher ---", force=True)
    
    if not os.path.exists(cfg.SCRIPT_SAMPLE_LDR):
        print(f"CRITICAL ERROR: Sampling script not found at: {cfg.SCRIPT_SAMPLE_LDR}")
        print("Please check SCRIPT_SAMPLE_LDR in project_config.py")
        sys.exit(1)

    # 2. Find Templates
    search_root = args.input_dir if args.input_dir else cfg.TEMPLATE_OUTPUT_DIR
    log(f"Scanning for templates in: {search_root}", force=True)
    
    # Recursive search for all .npz files
    template_paths = glob(os.path.join(search_root, "**", "*.npz"), recursive=True)
    template_paths.sort()

    if not template_paths:
        print(f"[!] No .npz files found in {search_root}")
        sys.exit(0)

    log(f"-> Found {len(template_paths)} templates to process.", force=True)

    # 3. Processing Loop
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for i, npz_path in enumerate(template_paths):
        
        # --- Prepare Paths & IDs ---
        fname = os.path.basename(npz_path).replace(".npz", "")
        # Parse: PROTID_idr_START-END
        match = re.match(r"([A-Z0-9]+)_idr", fname)
        if not match: 
            match = re.match(r"([A-Z0-9]+)_", fname)
            
        if not match:
            print(f"  [!] Skipping malformed filename: {fname}")
            continue

        prot_id = match.group(1)
        idr_label = fname.replace(f"{prot_id}_", "") # e.g., "idr_1-50"
        
        # Define specific output directory for this IDR
        # e.g. Pipeline_Outputs/Step_3_Raw_Conformers/P12345/idr_1-50/
        output_dir = os.path.join(cfg.CONFORMER_POOL_DIR, prot_id, idr_label)
        
        # --- Resume Logic ---
        # Check if we already have enough PDBs
        existing_pdbs = glob(os.path.join(output_dir, "*_relaxed.pdb"))
        if len(existing_pdbs) >= cfg.SAMPLE_N_CONFS:
            if not getattr(cfg, 'VERBOSE', True):
                 if i % 10 == 0: print(f"  ... Scanned {i}/{len(template_paths)} ...")
            else:
                 log(f"  [Skip] {prot_id} {idr_label}: {len(existing_pdbs)}/{cfg.SAMPLE_N_CONFS} done.")
            skipped_count += 1
            continue

        # --- Construct Command ---
        cmd = [
            cfg.PYTHON_EXEC, 
            cfg.SCRIPT_SAMPLE_LDR,
            cfg.MODEL_WEIGHTS_PATH,
            npz_path,           # fold_input
            output_dir,         # out_dir
            cfg.MODEL_CONFIG_PATH,
            "--batch", str(cfg.SAMPLE_BATCH_SIZE),
            "--nconf", str(cfg.SAMPLE_N_CONFS),
            "--ss_db", cfg.SS_DB_PATH
        ]

        # Optional flags
        if cfg.DEVICE == "cuda":
            cmd.append("--cuda")
        
        if getattr(cfg, 'ATTENTION_CHUNK_SIZE', 0) > 0:
            cmd.append("--attention_chunk")
            cmd.append(str(cfg.ATTENTION_CHUNK_SIZE))

        # --- Run Subprocess ---
        log(f"\n[{i+1}/{len(template_paths)}] Processing {prot_id} ({idr_label})...", force=True)
        
        try:
            result = subprocess.run(cmd, text=True)
            
            if result.returncode == 0:
                log(f"  -> Success.")
                processed_count += 1
            else:
                print(f"  [!] FAILURE: Subprocess returned code {result.returncode}")
                error_count += 1

        except KeyboardInterrupt:
            print("\n[!] User interrupted.")
            sys.exit(0)
        except Exception as e:
            print(f"  [!] Error launching subprocess: {e}")
            error_count += 1

    # 4. Summary
    log("\n--- Step 3 Batch Complete ---", force=True)
    log(f"Processed: {processed_count}", force=True)
    log(f"Skipped:   {skipped_count}", force=True)
    log(f"Errors:    {error_count}", force=True)

if __name__ == "__main__":
    main()