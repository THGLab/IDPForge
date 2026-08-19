"""Step 3: generate validated conformer pools per IDR via sample_ldr.py."""
import os
import sys
import subprocess
import argparse
import json
import shutil
import numpy as np
import glob
import re

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    import config as cfg
except ImportError as e:
    sys.exit(f"CRITICAL ERROR: Missing Dependency.\n{e}")

import json as _json_for_knots
from idpforge.utils.structure_validation import load_knot_screening, format_domain_spec

SEP = "-" * 60
STATE_FILENAME = ".step3_state.json"


# State persistence
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
        os.replace(tmp, state_path)
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


# Phase 1: generation + relaxation
def _parse_diffused(log_path):
    """Return the last '[LDR_STATS] diffused=N' count from a protein's log, or None."""
    n = None
    try:
        with open(log_path) as fh:
            for line in fh:
                m = re.search(r"\[LDR_STATS\] diffused=(\d+)", line)
                if m:
                    n = int(m.group(1))
    except OSError:
        pass
    return n


def generate_conformers(npz_path, output_dir, num_to_generate, verbose=False,
                        expected_knot_type=None, log_path=None):
    """Generate, relax, repair, and validate conformers via sample_ldr.py."""
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
        "--ss_db", cfg.SS_DB_PATH,
        "--fold_curv_ratio", str(cfg.SAMPLE_FOLD_CURV_RATIO),
        "--fold_curv_window", str(cfg.SAMPLE_FOLD_CURV_WINDOW),
        "--junction_kappa", str(cfg.SAMPLE_JUNCTION_KAPPA)
    ]
    if cfg.DEVICE == "cuda":
        cmd.append("--cuda")
    if verbose:
        cmd.append("--verbose")
    if expected_knot_type is not None:
        cmd.extend(["--expected_knot_type", _json_for_knots.dumps(expected_knot_type)])

    if verbose:
        print(f"   [Gen] Launching subprocess on GPU...", flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    if log_path:
        with open(log_path, "a", buffering=1) as fh:
            fh.write(f"\n{'='*60}\n[Round] target {target_total} "
                     f"(have {start_count}, generating {num_to_generate})\n")
            fh.flush()
            subprocess.run(cmd, check=True, env=env, stdout=fh, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True, env=env)

    # Clean up leftover raw files
    for raw_f in glob.glob(os.path.join(output_dir, "*_raw.pdb")):
        try:
            os.remove(raw_f)
        except OSError:
            pass

    # Collect validated files
    validated_files = sorted(
        glob.glob(os.path.join(output_dir, "*_validated.pdb")),
        key=lambda p: int(re.search(r'(\d+)_validated', os.path.basename(p)).group(1))
                       if re.search(r'(\d+)_validated', os.path.basename(p)) else 0
    )
    if verbose:
        print(f"   [Gen] {len(validated_files)} validated files now in directory.", flush=True)
    n_diffused = _parse_diffused(log_path) if log_path else None
    return validated_files, n_diffused


# Helpers
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


def _filter_knot_spec_for_window(expected_knot_type, sidecar):
    """Restrict a per-domain knot spec to a truncation window and shift into sliced numbering."""
    if sidecar is None or expected_knot_type is None:
        return expected_knot_type
    lo, hi, off = sidecar["trunc_lo"], sidecar["trunc_hi"], sidecar["offset"]
    out = []
    for d in expected_knot_type:
        rng = d.get("range")
        if not rng or len(rng) != 2:
            continue
        s, e = rng
        if s >= lo and e <= hi:
            nd = dict(d)
            nd["range"] = [s - off, e - off]
            out.append(nd)
    return out


# Main workflow per IDR
def run_idr_workflow(prot_id, npz_path, start_res, end_res, verbose=False,
                     expected_knot_type=None, sidecar_path=None, log_dir=None):
    range_tag = f"idr_{start_res}-{end_res}"
    output_dir = os.path.join(cfg.CONFORMER_POOL_DIR, prot_id, range_tag)
    os.makedirs(output_dir, exist_ok=True)
    # Per-protein live log
    log_path = os.path.join(log_dir, f"{prot_id}.log") if log_dir else None

    if sidecar_path and os.path.exists(sidecar_path):
        try:
            shutil.copyfile(sidecar_path, os.path.join(output_dir, "_truncation.json"))
        except Exception as e:
            print(f"   [Warning] could not copy truncation sidecar: {e}", flush=True)

    if log_path:
        print(f"\n   >>> Processing Region: {range_tag}   (live log: {log_path})", flush=True)
    else:
        print(f"\n   >>> Processing Region: {range_tag}", flush=True)

    # Load persisted state
    state = _load_state(output_dir, verbose=verbose)
    total_attempts = state["total_attempts"]

    # Resume: clean up orphaned files
    _cleanup_dir(output_dir)

    # Main loop
    while total_attempts < cfg.SAMPLE_MAX_TOTAL_ATTEMPTS:
        # 1. Check current validated count
        num_val, next_idx = _count_validated(output_dir)

        if num_val >= cfg.SAMPLE_N_CONFS:
            yield_pct = (100.0 * num_val / total_attempts) if total_attempts else 0.0
            print(f"   [Done] {range_tag}: {num_val}/{cfg.SAMPLE_N_CONFS} validated "
                  f"from {total_attempts} diffused ({yield_pct:.0f}% yield).", flush=True)
            _cleanup_dir(output_dir)
            break

        needed = cfg.SAMPLE_N_CONFS - num_val
        num_to_generate = needed
        if verbose:
            print(f"   [Status] Have {num_val}/{cfg.SAMPLE_N_CONFS}. Need {needed}. Generating {num_to_generate}.", flush=True)

        # 2. Generate + relax + validate conformers
        validated_files, n_diffused = generate_conformers(
            npz_path, output_dir, num_to_generate,
            verbose=verbose, expected_knot_type=expected_knot_type, log_path=log_path)

        new_valid = len(validated_files) - num_val
        total_attempts += n_diffused if n_diffused is not None else num_to_generate

        if new_valid == 0:
            print(f"   [Warning] No validated conformers produced. Retrying...", flush=True)

        _save_state(output_dir, total_attempts)

        if verbose:
            print(f"\n   [Round Summary] +{new_valid} validated this round. "
                  f"Total attempts: {total_attempts}.", flush=True)

    else:
        print(f"   [ABORT] Max attempts ({cfg.SAMPLE_MAX_TOTAL_ATTEMPTS}) reached "
              f"for {range_tag}.", flush=True)


# Main workflow per full IDP (Category 0)
def run_idp_workflow(prot_id, sequence, verbose=False, log_dir=None):
    """Category-0 full IDP: generate the ensemble directly with sample_idp.py."""
    output_dir = os.path.join(cfg.CONFORMER_POOL_DIR, prot_id)
    os.makedirs(output_dir, exist_ok=True)

    num_val, _ = _count_validated(output_dir)
    if num_val >= cfg.SAMPLE_N_CONFS:
        print(f"   [Done] Category-0 IDP already has {num_val}/{cfg.SAMPLE_N_CONFS}.", flush=True)
        return

    log_path = os.path.join(log_dir, f"{prot_id}.log") if log_dir else None
    live = f"   (live log: {log_path})" if log_path else ""
    print(f"   >>> Category-0 full IDP -> sample_idp.py "
          f"(have {num_val}/{cfg.SAMPLE_N_CONFS}, len={len(sequence)}){live}", flush=True)

    cmd = [
        cfg.PYTHON_EXEC, cfg.SCRIPT_SAMPLE_IDP,
        sequence, cfg.MODEL_WEIGHTS_PATH, output_dir, cfg.MODEL_CONFIG_PATH,
        "--batch", str(cfg.SAMPLE_BATCH_SIZE),
        "--nconf", str(cfg.SAMPLE_N_CONFS),
        "--ss_db", cfg.SS_DB_PATH,
    ]
    if cfg.DEVICE == "cuda":
        cmd.append("--cuda")
    if verbose:
        cmd.append("--verbose")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    if log_path:
        with open(log_path, "a", buffering=1) as fh:
            fh.write(f"\n{'='*60}\n[Category-0 IDP] target {cfg.SAMPLE_N_CONFS} "
                     f"(have {num_val}, len={len(sequence)})\n")
            fh.flush()
            subprocess.run(cmd, check=True, env=env, stdout=fh, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True, env=env)

    for raw_f in glob.glob(os.path.join(output_dir, "*_raw.pdb")):
        try:
            os.remove(raw_f)
        except OSError:
            pass


# Template discovery
def find_and_sort_templates(prot_id):
    search_pattern = f"{prot_id}_idr_*-*.npz"
    candidates = glob.glob(os.path.join(cfg.TEMPLATE_OUTPUT_DIR, "**", search_pattern), recursive=True)
    templates = []
    seen = set()
    for path in sorted(candidates):
        if "_parts" in path.replace("\\", "/").split("/"):
            continue
        match = re.search(r"_idr_(\d+)-(\d+)\.npz", os.path.basename(path))
        if not match:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        if key in seen:
            continue
        seen.add(key)
        sidecar_path = path[:-4] + ".trunc.json"
        templates.append({
            "path": path,
            "start": key[0],
            "end": key[1],
            "sidecar": sidecar_path if os.path.exists(sidecar_path) else None,
        })
    templates.sort(key=lambda x: x["start"])
    return templates


# Entry point
def main(args):
    verbose = args.verbose

    print("Step 3: Conformer Generation", flush=True)

    # Override config with CLI args
    cfg.TEMPLATE_OUTPUT_DIR = args.template_dir
    cfg.CONFORMER_POOL_DIR = args.output_dir
    cfg.SAMPLE_N_CONFS = args.n_confs
    cfg.SAMPLE_MAX_TOTAL_ATTEMPTS = args.max_attempts
    cfg.SAMPLE_BATCH_SIZE = args.batch_size
    cfg.DEVICE = args.device
    cfg.MODEL_WEIGHTS_PATH = args.weights
    cfg.MODEL_CONFIG_PATH = args.model_config
    cfg.SS_DB_PATH = args.ss_db
    cfg.SAMPLE_FOLD_CURV_RATIO = args.fold_curv_ratio
    cfg.SAMPLE_FOLD_CURV_WINDOW = args.fold_curv_window
    cfg.SAMPLE_JUNCTION_KAPPA = args.junction_kappa

    with open(args.id_file, 'r') as f:
        all_ids = sorted({l.strip() for l in f if l.strip()})

    my_chunk = all_ids
    if args.total_splits > 1:
        my_chunk = np.array_split(all_ids, args.total_splits)[args.split_index].tolist()

    # Per-protein live logs
    log_dir = os.path.join("logs", "Step3", "perprotein", f"split_{args.split_index}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Per-protein live logs: {log_dir}/<ID>.log  (tail -f to watch one)", flush=True)

    # Load knot screening
    knot_screening_path = getattr(cfg, 'KNOT_SCREENING_PATH',
        os.path.join(cfg.INPUT_DATA_DIR, "knot_screening.json"))
    knot_screening = load_knot_screening(knot_screening_path)
    if knot_screening:
        n_knotted = sum(
            1 for spec in knot_screening.values()
            if any(d.get("knot") for d in spec)
        )
        print(f"Knot screening: {len(knot_screening)} entries "
              f"({n_knotted} with at least one natively knotted domain).", flush=True)
    else:
        print(f"Knot screening: not available (full-chain unknot baseline will be applied).", flush=True)

    # Category-0 (full IDP) cases
    idp_cases = {}
    idp_cases_path = getattr(cfg, 'IDP_CASES_LIST_PATH', None)
    if idp_cases_path and os.path.isfile(idp_cases_path):
        try:
            with open(idp_cases_path) as f:
                idp_cases = json.load(f)
            print(f"Category-0 IDP cases: {len(idp_cases)} (full IDP via sample_idp.py).", flush=True)
        except Exception as e:
            print(f"[Warning] Could not load IDP cases ({idp_cases_path}): {e}", flush=True)

    print(f"Target conformers per IDR: {cfg.SAMPLE_N_CONFS}", flush=True)
    print(f"Max attempts per region: {cfg.SAMPLE_MAX_TOTAL_ATTEMPTS}", flush=True)

    for i, prot_id in enumerate(my_chunk):
        print(f"\n[{i+1}/{len(my_chunk)}] Protein: {prot_id}")

        # Category 0 (full IDP)
        if prot_id in idp_cases:
            run_idp_workflow(prot_id, idp_cases[prot_id], verbose=verbose, log_dir=log_dir)
            continue

        templates = find_and_sort_templates(prot_id)
        if not templates:
            print(f"   [SKIP] No templates found.", flush=True)
            continue

        # Expected per-domain knot spec
        expected_knot_type = knot_screening.get(prot_id) if knot_screening else None

        print(f"   Found {len(templates)} regions.", flush=True)
        if expected_knot_type is not None:
            print(f"   [Topology] Native spec: {format_domain_spec(expected_knot_type)}",
                  flush=True)
        for t in templates:
            sidecar = None
            if t.get("sidecar"):
                try:
                    with open(t["sidecar"]) as sf:
                        sidecar = json.load(sf)
                except Exception as e:
                    print(f"   [Warning] could not read sidecar {t['sidecar']}: {e}", flush=True)
            if sidecar is None and expected_knot_type:
                try:
                    _z = np.load(t["path"], allow_pickle=True)
                    if "graft_offset" in _z.files:
                        _off = int(_z["graft_offset"])
                        sidecar = {"trunc_lo": _off + 1,
                                   "trunc_hi": _off + int(len(_z["mask"])),
                                   "offset": _off}
                except Exception as e:
                    print(f"   [Warning] could not read graft_offset from {t['path']}: {e}", flush=True)
            local_knot = _filter_knot_spec_for_window(expected_knot_type, sidecar)
            if sidecar is not None and expected_knot_type is not None and verbose:
                print(f"   [Topology] idr {t['start']}-{t['end']} (truncated): "
                      f"{format_domain_spec(local_knot)}", flush=True)
            run_idr_workflow(prot_id, t["path"], t["start"], t["end"],
                            verbose=verbose, expected_knot_type=local_knot,
                            sidecar_path=t.get("sidecar"), log_dir=log_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Phased Conformer Generation Pipeline")
    parser.add_argument("--id_file", default=cfg.RUN_ID_FILE,
                        help=f"Newline-separated UniProt ID list (default from config: {cfg.RUN_ID_FILE}).")
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
    parser.add_argument("--fold_curv_ratio", type=float, default=cfg.SAMPLE_FOLD_CURV_RATIO,
                        help=f"Folded-domain curvature gate vs the template (fraction of template curvature); "
                             f"0 disables (default: {cfg.SAMPLE_FOLD_CURV_RATIO}).")
    parser.add_argument("--fold_curv_window", type=int, default=cfg.SAMPLE_FOLD_CURV_WINDOW,
                        help=f"Folded residues into the fold from each junction to average for the curvature "
                             f"gate (default: {cfg.SAMPLE_FOLD_CURV_WINDOW}).")
    parser.add_argument("--junction_kappa", type=float, default=cfg.SAMPLE_JUNCTION_KAPPA,
                        help=f"Junction-IDR curvature gate (taut-reach filter), A^-1; 0 disables (default: {cfg.SAMPLE_JUNCTION_KAPPA}).")
    parser.add_argument("--verbose", action="store_true", default=cfg.VERBOSE,
                        help="Enable detailed per-conformer logging.")
    args = parser.parse_args()
    main(args)
