"""
Step 3: Conformer Generation Pipeline
1. Checks validated count.
2. If deficit, generates conformers via sample_ldr.py (handles relaxation,
   repair, and validation internally via output_to_pdb).
3. Repeats until target is met or max attempts exhausted.
"""
import os
import sys
import subprocess
import argparse
import json
import numpy as np
import glob
import re

# ------------------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    import config as cfg
except ImportError as e:
    sys.exit(f"CRITICAL ERROR: Missing Dependency.\n{e}")

SEP = "-" * 60
STATE_FILENAME = ".step3_state.json"


# ------------------------------------------------------------------------
# STATE PERSISTENCE (for cluster resumability)
# ------------------------------------------------------------------------
def _load_state(output_dir, verbose=False):
    """Load persisted state from a previous run, or return defaults."""
    state_path = os.path.join(output_dir, STATE_FILENAME)
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            if verbose:
                print(f"   [Resume] Loaded state: {state['total_attempts']} prior attempts.", flush=True)
            return state
        except Exception:
            pass
    return {"total_attempts": 0}


def _save_state(output_dir, total_attempts):
    """Persist state to disk so runs can be resumed."""
    state_path = os.path.join(output_dir, STATE_FILENAME)
    state = {"total_attempts": total_attempts}
    try:
        tmp = state_path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(state, f)
        os.replace(tmp, state_path)  # atomic on POSIX
    except Exception as e:
        print(f"   [Warning] Could not save state: {e}", flush=True)


def _cleanup_dir(output_dir):
    """Remove leftover raw and relaxed files (but not validated or state)."""
    for pattern in ("*_raw.pdb", "*_relaxed.pdb", ".tmp_*"):
        for f in glob.glob(os.path.join(output_dir, pattern)):
            try:
                os.remove(f)
            except OSError:
                pass


# ------------------------------------------------------------------------
# PHASE 1: GENERATION + RELAXATION (subprocess to sample_ldr.py)
# ------------------------------------------------------------------------
def generate_conformers(npz_path, output_dir, num_to_generate, verbose=False):
    """
    Calls sample_ldr.py to generate, relax, repair, and validate conformers.
    Returns list of validated file paths.
    """
    # Count existing validated files to determine starting counter
    existing = glob.glob(os.path.join(output_dir, "*_validated.pdb"))
    start_count = len(existing)
    target_total = start_count + num_to_generate

    if verbose:
        print(f"   [Gen] Generating {num_to_generate} new conformers (with relaxation)...", flush=True)

    cmd = [
        cfg.PYTHON_EXEC, cfg.SCRIPT_SAMPLE_LDR,
        cfg.MODEL_WEIGHTS_PATH, npz_path, output_dir, cfg.MODEL_CONFIG_PATH,
        "--batch", str(cfg.SAMPLE_BATCH_SIZE),
        "--nconf", str(target_total),
        "--ss_db", cfg.SS_DB_PATH
    ]
    if cfg.DEVICE == "cuda":
        cmd.append("--cuda")
    if verbose:
        cmd.append("--verbose")

    if verbose:
        print(f"   [Gen] Launching subprocess on GPU...", flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    subprocess.run(cmd, check=True, env=env)

    # Clean up leftover raw files (sample_ldr writes raw then relaxes, but
    # does not delete the raw files itself)
    for raw_f in glob.glob(os.path.join(output_dir, "*_raw.pdb")):
        try:
            os.remove(raw_f)
        except OSError:
            pass

    # Collect all validated files (sample_ldr now handles repair + validation)
    validated_files = sorted(
        glob.glob(os.path.join(output_dir, "*_validated.pdb")),
        key=lambda p: int(re.search(r'(\d+)_validated', os.path.basename(p)).group(1))
                       if re.search(r'(\d+)_validated', os.path.basename(p)) else 0
    )
    if verbose:
        print(f"   [Gen] {len(validated_files)} validated files now in directory.", flush=True)
    return validated_files


# ------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------
def _count_validated(output_dir):
    """Count existing validated files and return (count, next_index)."""
    v_files = glob.glob(os.path.join(output_dir, "*_validated.pdb"))
    if not v_files:
        return 0, 1
    indices = []
    for f in v_files:
        m = re.search(r'(\d+)_validated', os.path.basename(f))
        if m:
            indices.append(int(m.group(1)))
    return len(v_files), (max(indices) + 1 if indices else 1)


# ------------------------------------------------------------------------
# MAIN WORKFLOW PER IDR
# ------------------------------------------------------------------------
def run_idr_workflow(prot_id, npz_path, start_res, end_res, verbose=False):
    range_tag = f"idr_{start_res}-{end_res}"
    output_dir = os.path.join(cfg.CONFORMER_POOL_DIR, prot_id, range_tag)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n   >>> Processing Region: {range_tag}", flush=True)

    # Load persisted state (survives cluster preemption / restarts)
    state = _load_state(output_dir, verbose=verbose)
    total_attempts = state["total_attempts"]

    # ------------------------------------------------------------------
    # RESUME PHASE: Clean up orphaned files from a killed run
    # ------------------------------------------------------------------
    # Relaxed files without a matching validated file are leftovers from
    # interrupted runs. Since sample_ldr now handles repair+validation
    # internally, these are safely discarded.
    _cleanup_dir(output_dir)

    # ------------------------------------------------------------------
    # MAIN LOOP: Generate conformers (sample_ldr handles relax + validate)
    # ------------------------------------------------------------------
    while total_attempts < cfg.SAMPLE_MAX_TOTAL_ATTEMPTS:
        # 1. Check current validated count
        num_val, next_idx = _count_validated(output_dir)

        if num_val >= cfg.SAMPLE_N_CONFS:
            print(f"   [Done] Target reached ({num_val}/{cfg.SAMPLE_N_CONFS}).", flush=True)
            _cleanup_dir(output_dir)
            break

        needed = cfg.SAMPLE_N_CONFS - num_val
        num_to_generate = needed
        if verbose:
            print(f"   [Status] Have {num_val}/{cfg.SAMPLE_N_CONFS}. Need {needed}. Generating {num_to_generate}.", flush=True)

        # 2. Generate + relax + validate conformers (sample_ldr handles all via output_to_pdb)
        validated_files = generate_conformers(npz_path, output_dir, num_to_generate, verbose=verbose)

        new_valid = len(validated_files) - num_val
        total_attempts += num_to_generate

        if new_valid == 0:
            print(f"   [Warning] No validated conformers produced. Retrying...", flush=True)

        # Persist state after every round
        _save_state(output_dir, total_attempts)

        if verbose:
            print(f"\n   [Round Summary] +{new_valid} validated this round. "
                  f"Total attempts: {total_attempts}.", flush=True)

    else:
        print(f"   [ABORT] Max attempts ({cfg.SAMPLE_MAX_TOTAL_ATTEMPTS}) reached "
              f"for {range_tag}.", flush=True)


# ------------------------------------------------------------------------
# TEMPLATE DISCOVERY
# ------------------------------------------------------------------------
def find_and_sort_templates(prot_id):
    search_pattern = f"{prot_id}_idr_*-*.npz"
    candidates = glob.glob(os.path.join(cfg.TEMPLATE_OUTPUT_DIR, "**", search_pattern), recursive=True)
    templates = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        fname = os.path.basename(path)
        match = re.search(r"_idr_(\d+)-(\d+)\.npz", fname)
        if match:
            templates.append({"path": path, "start": int(match.group(1)), "end": int(match.group(2))})
    templates.sort(key=lambda x: x["start"])
    return templates


# ------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------
def main(args):
    verbose = args.verbose

    print("=" * 60, flush=True)
    print("  Phased Conformer Generation Pipeline (Step 3)", flush=True)
    print("=" * 60, flush=True)

    # Override config with CLI args so all helper functions pick them up
    cfg.TEMPLATE_OUTPUT_DIR = args.template_dir
    cfg.CONFORMER_POOL_DIR = args.output_dir
    cfg.SAMPLE_N_CONFS = args.n_confs
    cfg.SAMPLE_MAX_TOTAL_ATTEMPTS = args.max_attempts
    cfg.SAMPLE_BATCH_SIZE = args.batch_size
    cfg.DEVICE = args.device
    cfg.MODEL_WEIGHTS_PATH = args.weights
    cfg.MODEL_CONFIG_PATH = args.model_config
    cfg.SS_DB_PATH = args.ss_db

    with open(args.id_file, 'r') as f:
        all_ids = sorted({l.strip() for l in f if l.strip()})

    my_chunk = all_ids
    if args.total_splits > 1:
        my_chunk = np.array_split(all_ids, args.total_splits)[args.split_index].tolist()

    print(f"Job {args.split_index+1}/{args.total_splits}: Processing {len(my_chunk)} Proteins", flush=True)
    print(f"Target: {cfg.SAMPLE_N_CONFS} conformers per IDR | "
          f"Max attempts: {cfg.SAMPLE_MAX_TOTAL_ATTEMPTS}", flush=True)

    for i, prot_id in enumerate(my_chunk):
        print(f"\n[{i+1}/{len(my_chunk)}] Protein: {prot_id}")
        templates = find_and_sort_templates(prot_id)

        if not templates:
            print(f"   [SKIP] No templates found.", flush=True)
            continue

        print(f"   Found {len(templates)} regions.", flush=True)
        for t in templates:
            run_idr_workflow(prot_id, t["path"], t["start"], t["end"], verbose=verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Phased Conformer Generation Pipeline")
    parser.add_argument("--id_file", required=True,
                        help="Path to a text file containing newline-separated UniProt IDs.")
    parser.add_argument("--total_splits", type=int, default=1,
                        help="Total number of parallel jobs (default: 1).")
    parser.add_argument("--split_index", type=int, default=0,
                        help="The specific shard index, 0-based (default: 0).")
    parser.add_argument("--template_dir", default=cfg.TEMPLATE_OUTPUT_DIR,
                        help=f"Directory containing Step 2 .npz templates (default: {cfg.TEMPLATE_OUTPUT_DIR}).")
    parser.add_argument("--output_dir", default=cfg.CONFORMER_POOL_DIR,
                        help=f"Output directory for generated conformers (default: {cfg.CONFORMER_POOL_DIR}).")
    parser.add_argument("--n_confs", type=int, default=cfg.SAMPLE_N_CONFS,
                        help=f"Target number of validated conformers per IDR (default: {cfg.SAMPLE_N_CONFS}).")
    parser.add_argument("--max_attempts", type=int, default=cfg.SAMPLE_MAX_TOTAL_ATTEMPTS,
                        help=f"Maximum total generation attempts per IDR (default: {cfg.SAMPLE_MAX_TOTAL_ATTEMPTS}).")
    parser.add_argument("--batch_size", type=int, default=cfg.SAMPLE_BATCH_SIZE,
                        help=f"Batch size for diffusion sampling (default: {cfg.SAMPLE_BATCH_SIZE}).")
    parser.add_argument("--device", default=cfg.DEVICE,
                        help=f"Device for inference: 'cuda' or 'cpu' (default: {cfg.DEVICE}).")
    parser.add_argument("--weights", default=cfg.MODEL_WEIGHTS_PATH,
                        help=f"Path to model weights checkpoint (default: {cfg.MODEL_WEIGHTS_PATH}).")
    parser.add_argument("--model_config", default=cfg.MODEL_CONFIG_PATH,
                        help=f"Path to model YAML config (default: {cfg.MODEL_CONFIG_PATH}).")
    parser.add_argument("--ss_db", default=cfg.SS_DB_PATH,
                        help=f"Path to secondary structure database (default: {cfg.SS_DB_PATH}).")
    parser.add_argument("--verbose", action="store_true", default=cfg.VERBOSE,
                        help="Enable detailed per-conformer logging.")
    args = parser.parse_args()
    main(args)
