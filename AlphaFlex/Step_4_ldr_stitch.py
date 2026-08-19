"""Step 4: stitch pre-generated IDR ensembles onto folded domains into full-length models."""
import os
import glob
import re
import argparse
import sys
from collections import Counter
import numpy as np
import json
import io
import shutil
from tqdm import tqdm

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import contextlib


def _process_protein_to_file(protein_id, _log_dir, **kwargs):
    """Run process_protein with its stdout/stderr streamed to a per-protein log file."""
    log_path = os.path.join(_log_dir, f"{protein_id}.log")
    result, err = None, None
    with open(log_path, "w", buffering=1) as fh:
        with contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
            try:
                result = process_protein(protein_id=protein_id, **kwargs)
            except BaseException:
                import traceback
                traceback.print_exc()
                err = traceback.format_exc()
    return result, log_path, err


def mkensemble(pdb_files):
    """Concatenate pre-stamped single-MODEL conformer PDBs into one ensemble."""
    for path in pdb_files:
        with open(path) as fh:
            for line in fh:
                if line.startswith(("PARENT", "REMARK", "MASTER", "CRYST")):
                    continue
                if line.rstrip() == "END":
                    continue
                yield line if line.endswith("\n") else line + "\n"
    yield "END\n"



def _max_ensemble_n(final_dest_dir, protein_id):
    """Largest N among existing ``<protein>_ensemble_nN.pdb`` files (-1 if none)."""
    best = -1
    for p in glob.glob(os.path.join(final_dest_dir, f"{protein_id}_ensemble_n*.pdb")):
        m = re.search(r"_ensemble_n(\d+)\.pdb$", os.path.basename(p))
        if m:
            best = max(best, int(m.group(1)))
    return best


def _clear_stale_ensembles(final_dest_dir, protein_id):
    """Remove existing ``<protein>_ensemble_nN.pdb`` files."""
    for p in glob.glob(os.path.join(final_dest_dir, f"{protein_id}_ensemble_n*.pdb")):
        try:
            os.remove(p)
        except OSError:
            pass


# --- BioPython Setup ---
try:
    from Bio.PDB import PDBParser, PDBIO
    print("BioPython.PDB loaded successfully.")
except ImportError:
    print("Error: BioPython is required. Run: conda install -c conda-forge biopython")
    sys.exit(1)

# --- OpenMM Setup ---
try:
    from openmm.app import PDBFile
    print("OpenMM loaded successfully.")
except ImportError:
    print("Error: OpenMM is required for this script.")
    print("Please install it in your environment: conda install -c conda-forge openmm")
    sys.exit(1)

# --- Imports ---
from idpforge.utils.relax import relax_protein
from idpforge.misc import _ca_from_pdb_file, finalize_model_pdb
from idpforge.utils.structure_validation import _check_fold_curvature

# --- Import Config ---
try:
    import config as cfg
except ImportError:
    print("Error: auto_config.py not found. Please create it.")
    sys.exit(1)

# --- Shared Utility Imports ---
from utils.smart_scoring import get_smart_threshold
from idpforge.utils.structure_repair import repair_chirality, fix_histidine_naming
from idpforge.utils.structure_validation import (
    validate_structure_post_relax, check_bond_integrity, load_knot_screening,
    format_domain_spec
)

# --- Stitch Utility Imports ---
from utils.stitch import (
    get_completion_status, get_length_label, build_region_resids,
    get_id_to_pdb_path, get_protein_category, find_ensemble_dirs,
    format_ranges, load_pdb_structure, get_segment_atoms,
    build_segment_map, assemble_kinematic_chain, clean_structure,
    renumber_pool
)
from utils import graft_back


# --- Energy Minimization with Amber ---
relax_cfg = {
    'max_outer_iterations': cfg.RELAX_MAX_OUTER_ITER,
    'stiffness': cfg.RELAX_STIFFNESS,
    'exclude_residues': [],
    'max_iterations': cfg.MINIMIZATION_MAX_ITER,
    'tolerance': cfg.MINIMIZATION_TOLERANCE
}

def relax_with_established_method(structure, output_filepath, idr_indices=None, device="cuda:0", verbose=False):
    """Energy-minimize the model with the AMBER99SB forcefield (folded domains restrained)."""
    pdb_name = os.path.splitext(os.path.basename(output_filepath))[0]
    output_dir = os.path.dirname(output_filepath)

    # 1. Setup config
    run_config = relax_cfg.copy()
    if idr_indices:
        run_config['exclude_residues'] = idr_indices

    # 2. Convert BioPython Structure -> PDB String
    io_pdb = PDBIO()
    io_pdb.set_structure(structure)
    buf = io.StringIO()
    io_pdb.save(buf)
    pdb_str = buf.getvalue()

    # 3. Convert to OpenFold Object
    try:
        from openfold.np import protein as of_protein
        unrelaxed_prot = of_protein.from_pdb_string(pdb_str)
    except Exception as e:
        if verbose:
            print(f"       [Error] PDB parsing failed: {e}")
        return False

    # 4. Run Relaxation
    try:
        n_res = len(unrelaxed_prot.aatype)
        if idr_indices:
            viol_mask = np.zeros(n_res, dtype=bool)
            idx = np.asarray(idr_indices, dtype=int)
            viol_mask[idx[(idx >= 0) & (idx < n_res)]] = True
        else:
            viol_mask = None
        result = relax_protein(
            config=run_config,
            model_device=device,
            unrelaxed_protein=unrelaxed_prot,
            output_dir=output_dir,
            pdb_name=pdb_name,
            viol_threshold=0.02,
            viol_mask=viol_mask
        )

        expected_output = os.path.join(output_dir, f"{pdb_name}_relaxed.pdb")

        if result == 1 and os.path.exists(expected_output):
            if os.path.exists(output_filepath): os.remove(output_filepath)
            os.rename(expected_output, output_filepath)
            return True
        else:
            if verbose:
                print("       [FAIL] Relaxation rejected.")
            if os.path.exists(expected_output): os.remove(expected_output)
            return False

    except Exception as e:
        if verbose:
            print(f"       [CRASH] Relaxation failed: {e}")
        return False

# --- Truncated-pool back-numbering ---
def _maybe_renumber_truncated_pool(pool_path, dest_dir, label, verbose=False):
    """Map a truncated sub-ensemble back to full-length numbering before stitching."""
    sidecar = os.path.join(pool_path, "_truncation.json")
    if not os.path.exists(sidecar):
        return pool_path, None
    try:
        with open(sidecar) as sf:
            offset = int(json.load(sf).get("offset", 0))
    except Exception as e:
        if verbose:
            print(f"  [TRUNC] {label}: unreadable sidecar {sidecar} ({e}); using pool as-is.")
        return pool_path, None
    if offset == 0:
        return pool_path, None
    renum_dir = os.path.join(dest_dir, f"_renum_{label}")
    new_files = renumber_pool(pool_path, renum_dir, offset, verbose=verbose)
    if not new_files:
        if verbose:
            print(f"  [TRUNC] {label}: sidecar offset {offset} but no conformers to renumber.")
        return pool_path, None
    if verbose:
        print(f"  [TRUNC] {label}: renumbered {len(new_files)} conformers +{offset} -> full-length.")
    return renum_dir, new_files

_HIS_RESNAMES = {'HIS', 'HID', 'HIE', 'HIP'}


# --- Shared post-stitch relax + validation ---
def _relax_repair_validate(raw, out_path, region_resids, gate_ctx, expected_knot_type,
                           attempts, done, pdb_parser, verbose=False, device="cuda:0"):
    """Relax, repair, re-relax, validate, and fold-gate one assembled full-length model."""
    # Identify IDR vs folded residues
    idr_idx, frozen_ids, cnt = [], [], 0
    for r in sorted(raw[0].get_residues(), key=lambda x: x.id[1]):
        if r.id[0] == ' ':
            if r.id[1] in region_resids:
                idr_idx.append(cnt)
            else:
                frozen_ids.append(r.id[1])
            cnt += 1
    if verbose:
        print(f"       [CONFIG] Freezing residues: {format_ranges(frozen_ids)}")

    # Pre-minimization: save + chirality repair
    io_save = PDBIO()
    io_save.set_structure(raw)
    io_save.save(out_path)
    repair_chirality(out_path, verbose=False)
    repaired = load_pdb_structure(out_path, pdb_parser, verbose=verbose) or raw

    # Relaxation
    if not relax_with_established_method(repaired, out_path, idr_indices=idr_idx, device=device, verbose=verbose):
        if verbose:
            print(f"       [RESULT] FAILED (Relaxation Rejected)")
        if os.path.exists(out_path):
            os.remove(out_path)
        return False, {"reason": "Relaxation Rejected", "_threshold": 0.0}

    # Post-relax repair + re-relax
    needs_rerelax = False
    try:
        chk_pdb = PDBFile(out_path)
        broken = check_bond_integrity(chk_pdb.topology, chk_pdb.positions)
        his_resids = {b['resid'] for b in broken
                      if b['resname'] in _HIS_RESNAMES or b.get('resname2', '') in _HIS_RESNAMES}
    except Exception:
        his_resids = set()

    n_chiral = repair_chirality(out_path, verbose=False)
    if n_chiral > 0:
        needs_rerelax = True
        if verbose:
            print(f"       [REPAIR] Flipped {n_chiral} D-isomer(s).")

    if his_resids:
        try:
            n_his = fix_histidine_naming(out_path, his_resids, verbose=False)
            if n_his and n_his > 0:
                needs_rerelax = True
                if verbose:
                    print(f"       [REPAIR] Fixed {n_his} HIS naming(s).")
        except Exception as e:
            if verbose:
                print(f"       [ERROR] HIS naming fix failed: {e}")

    if needs_rerelax:
        rep2 = load_pdb_structure(out_path, pdb_parser, verbose=verbose)
        if not rep2 or not relax_with_established_method(
                rep2, out_path, idr_indices=idr_idx, device=device, verbose=verbose):
            if verbose:
                print(f"       [RESULT] FAILED (Re-relax after repair)")
            if os.path.exists(out_path):
                os.remove(out_path)
            return False, {"reason": "Re-relax after repair", "_threshold": 0.0}

    # --- Post-minimization validation ---
    if verbose:
        print(f"       [POST-MIN CHECK] Validating...")
    chk = PDBFile(out_path)
    threshold = get_smart_threshold(
        attempts, done, base=cfg.STITCH_BASE_CLASH_THRESHOLD, inc=cfg.STITCH_CLASH_INCREMENT)
    idr_start_res = min(region_resids) if region_resids else None
    idr_end_res = max(region_resids) if region_resids else None

    is_valid, info = validate_structure_post_relax(
        chk.topology, chk.positions, pdb_path=out_path,
        strict_clash_threshold=threshold, idr_start=idr_start_res, idr_end=idr_end_res,
        verbose=False, full_report=True, expected_knot_type=expected_knot_type)
    info["_threshold"] = threshold

    if verbose:
        chiral_str = "PASS" if info.get("chirality_pass", True) else "FAIL"
        bonds_str = "PASS" if info.get("bonds_pass", True) else f"FAIL ({info.get('num_broken_bonds', 0)} broken)"
        clash_str = "PASS" if info.get("clash_pass", True) else "FAIL"
        if expected_knot_type is not None:
            detected = info.get("detected_knot_type")
            disp = info.get("expected_knot_display") or format_domain_spec(expected_knot_type)
            if info.get("knot_pass", True):
                knot_str = f"PASS (native {disp})"
            else:
                knot_str = f"FAIL (expected {disp}, got {detected}) [{info.get('knot_fail_reason', '')}]"
        else:
            knot_str = "PASS" if info.get("knot_pass", True) else f"FAIL ({info.get('knot_type')})"
        print(f"         - Chirality: {chiral_str}")
        print(f"         - Bonds:     {bonds_str}")
        print(f"         - Clashes:   {info.get('num_clashes', '?')} (Score: {info.get('clash_score', 0):.2f} | Limit: {threshold:.1f}) -> {clash_str}")
        print(f"         - Topology:  {knot_str}")

    # --- Folded-domain curvature gate ---
    if is_valid and gate_ctx is not None:
        if cnt != gate_ctx["n_res"]:
            is_valid = False
            info["reason"] = info.get("reason", "") + f" | fold-gate residue count {cnt}!={gate_ctx['n_res']}"
            if verbose:
                print(f"         - Fold curvature: FAIL (residue count {cnt}!={gate_ctx['n_res']})")
        else:
            model_ca = _ca_from_pdb_file(out_path, gate_ctx["n_res"])
            gate_pass, gate_lines, gate_reason = _check_fold_curvature(
                model_ca, gate_ctx["ref_ca"], gate_ctx["folded_mask"],
                gate_ctx["curv_ratio"], window=gate_ctx["curv_window"])
            if not gate_pass:
                is_valid = False
                info["reason"] = info.get("reason", "") + " | " + gate_reason
            if verbose:
                detail = f"  ({gate_lines[0]})" if gate_lines else ""
                print(f"         - Fold curvature: {'PASS' if gate_pass else 'FAIL'}{detail}")

    if not is_valid and os.path.exists(out_path):
        os.remove(out_path)
    return is_valid, info


# --- Truncated-pool graft reconstruction ---
def _load_graft_spec(pool_dir):
    """Read a `_truncation.json` graft spec, or None if it is not a graft pool."""
    sc = os.path.join(pool_dir, "_truncation.json")
    if not os.path.exists(sc):
        return None
    try:
        with open(sc) as f:
            d = json.load(f)
    except Exception:
        return None
    if "idr_ranges" not in d:
        return None
    if "fold_ranges" in d:
        franges = [tuple(x) for x in d["fold_ranges"]]
    elif "fold_range" in d:
        franges = [tuple(d["fold_range"])]
    else:
        return None
    if not franges:
        return None
    return {"offset": int(d.get("offset", 0)),
            "idr_ranges": [tuple(x) for x in d["idr_ranges"]],
            "fold_ranges": franges,
            "fold_range": (min(lo for lo, _ in franges), max(hi for _, hi in franges))}


def _build_fold_gate(static_path, model_resseqs, idr_ranges, curv_ratio, curv_window):
    """Build the folded-domain curvature gate context for a reconstruction's residues."""
    if curv_ratio <= 0:
        return None
    ca = graft_back._ca_by_resseq(graft_back.parse_pdb_atoms(static_path), set(model_resseqs))
    ref = np.zeros((len(model_resseqs), 3))
    mask = np.zeros(len(model_resseqs), dtype=bool)
    for k, rs in enumerate(model_resseqs):
        if rs in ca:
            ref[k] = ca[rs]
        mask[k] = not any(lo <= rs <= hi for (lo, hi) in idr_ranges)
    return {"ref_ca": ref, "folded_mask": mask, "n_res": len(model_resseqs),
            "curv_ratio": curv_ratio, "curv_window": curv_window}


def reconstruct_truncated_pool(protein_id, pool_dir, spec, static_path, num_conformers,
                               work_dir, final_dest_dir, knot_screening=None, verbose=False,
                               device="cuda:0"):
    """Finalize a size-capped single-IDR pool into a validated full-length chimera ensemble."""
    pdb_parser = PDBParser(QUIET=True)
    static_atoms = graft_back.parse_pdb_atoms(static_path)
    flo, fhi = spec["fold_range"]
    idr_ranges, offset = spec["idr_ranges"], spec["offset"]

    region_resids = set()
    for lo, hi in idr_ranges:
        region_resids.update(range(lo, hi + 1))
    static_fold_present = {a["resseq"] for a in static_atoms if flo <= a["resseq"] <= fhi}
    model_resseqs = sorted(region_resids | static_fold_present)

    curv_ratio = getattr(cfg, 'STITCH_FOLD_CURV_RATIO', 0.0) or 0.0
    curv_window = int(getattr(cfg, 'STITCH_FOLD_CURV_WINDOW', 15))
    gate_ctx = _build_fold_gate(static_path, model_resseqs, idr_ranges, curv_ratio, curv_window)
    expected_knot_type = knot_screening.get(protein_id) if knot_screening else None

    os.makedirs(work_dir, exist_ok=True)
    # Clear stale accepted conformers
    for _stale in glob.glob(os.path.join(work_dir, "minimized_conformer_*.pdb")):
        os.remove(_stale)

    def _cidx(p):
        m = re.search(r'(\d+)_validated', os.path.basename(p))
        return (0, int(m.group(1))) if m else (1, os.path.basename(p))
    conformers = sorted(glob.glob(os.path.join(pool_dir, "*_validated.pdb")), key=_cidx)

    if verbose:
        print(f"  [Reconstruct] {protein_id}: {len(conformers)} truncated conformers -> graft "
              f"(fold {flo}-{fhi}, IDR {idr_ranges}, offset {offset}) + relax + validate.")

    import random
    rng = random.Random(getattr(cfg, 'STITCH_PAIR_SEED', 0))
    done, attempts = 0, 0

    def _attempt(conf):
        """Graft one truncated conformer's fold back, relax, and validate."""
        nonlocal done, attempts
        attempts += 1
        grafted = os.path.join(work_dir, f"_grafted_{attempts}.pdb")
        rep = graft_back.graft_conformer_multi(
            conf, static_atoms, idr_ranges, spec["fold_ranges"], offset, grafted)
        if rep is None:
            if verbose:
                print(f"     [Attempt {attempts}] graft failed (insufficient fold overlap); skipping.")
            if os.path.exists(grafted):
                os.remove(grafted)
            return
        raw = load_pdb_structure(grafted, pdb_parser, verbose=verbose)
        if os.path.exists(grafted):
            os.remove(grafted)
        if not raw:
            return
        out_path = os.path.join(work_dir, f"minimized_conformer_{done + 1}.pdb")
        if verbose:
            print(f"\n{'-'*60}\n     [Attempt {attempts}] Grafted -> full length "
                  f"(junction {rep['junction_gap']}, seam {rep['seam_gap']})")
        try:
            is_valid, info = _relax_repair_validate(
                raw, out_path, region_resids, gate_ctx, expected_knot_type,
                attempts, done, pdb_parser, verbose=verbose, device=device)
        except Exception as e:
            if verbose:
                print(f"       [ERROR] Validation crashed: {e}")
            if os.path.exists(out_path):
                os.remove(out_path)
            return
        if is_valid:
            done += 1
            finalize_model_pdb(out_path, done)
            if verbose:
                print(f"       [RESULT] SUCCESS! (Total: {done}/{num_conformers})")
        elif verbose:
            print(f"       [RESULT] FAILED ({info.get('reason', 'Unknown')}) "
                  f"[Thresh: {info.get('_threshold', 0):.2f}]")

    # Phase 1: single pass
    for conf in conformers:
        if done >= num_conformers:
            break
        _attempt(conf)

    # Phase 2: top-up
    max_attempts = getattr(cfg, 'STITCH_MAX_ATTEMPTS', num_conformers * 5)
    if done < num_conformers and conformers:
        if verbose:
            print(f"    [Top-up] single pass gave {done}/{num_conformers}; filling the rest by "
                  f"random sampling WITH replacement (cap {max_attempts} attempts).", flush=True)
        while done < num_conformers and attempts < max_attempts:
            _attempt(rng.choice(conformers))
        if verbose and done < num_conformers:
            print(f"    [Note] reached {done}/{num_conformers} after {attempts} attempts "
                  f"(hit the {max_attempts}-attempt cap).", flush=True)

    accepted = sorted(glob.glob(os.path.join(work_dir, "minimized_conformer_*.pdb")),
                      key=lambda p: int(re.search(r'(\d+)\.pdb', p).group(1)))
    if not accepted:
        if verbose:
            print(f"    [!] No conformers survived reconstruction for {protein_id}; no ensemble written.")
        return "Failed", 0
    os.makedirs(final_dest_dir, exist_ok=True)
    _clear_stale_ensembles(final_dest_dir, protein_id)
    ensemble_path = os.path.join(final_dest_dir, f"{protein_id}_ensemble_n{len(accepted)}.pdb")
    with open(ensemble_path, 'w') as f:
        f.writelines(mkensemble(accepted))
    if verbose:
        print(f"    -> Reconstructed + validated ensemble: {ensemble_path} ({len(accepted)} models)")
    return get_completion_status(len(accepted), num_conformers), len(accepted)


def reconstruct_combined_chimera(protein_id, tail_specs, static_path, fold_range, num_conformers,
                                 work_dir, final_dest_dir, knot_screening=None, verbose=False,
                                 device="cuda:0", max_attempts=None):
    """Assemble and validate a multi-tail full-length chimera ensemble."""
    import random
    pdb_parser = PDBParser(QUIET=True)
    static_atoms = graft_back.parse_pdb_atoms(static_path)
    flo, fhi = fold_range

    idr_ranges = [tuple(ts["idr_range"]) for ts in tail_specs]
    region_resids = set()
    for lo, hi in idr_ranges:
        region_resids.update(range(lo, hi + 1))
    static_fold_present = {a["resseq"] for a in static_atoms if flo <= a["resseq"] <= fhi}
    model_resseqs = sorted(region_resids | static_fold_present)

    curv_ratio = getattr(cfg, 'STITCH_FOLD_CURV_RATIO', 0.0) or 0.0
    curv_window = int(getattr(cfg, 'STITCH_FOLD_CURV_WINDOW', 15))
    gate_ctx = _build_fold_gate(static_path, model_resseqs, idr_ranges, curv_ratio, curv_window)
    expected_knot_type = knot_screening.get(protein_id) if knot_screening else None

    def _cidx(p):
        m = re.search(r'(\d+)_validated', os.path.basename(p))
        return int(m.group(1)) if m else (1 << 30)
    tail_files = [sorted(glob.glob(os.path.join(ts["pool_dir"], "*_validated.pdb")), key=_cidx)
                  for ts in tail_specs]
    if any(len(fs) == 0 for fs in tail_files):
        return "Failed", 0

    os.makedirs(work_dir, exist_ok=True)
    for _stale in glob.glob(os.path.join(work_dir, "stitched_conformer_*.pdb")):
        os.remove(_stale)

    rng = random.Random(getattr(cfg, 'STITCH_PAIR_SEED', 0))
    _pools = []
    for fs in tail_files:
        fs = list(fs)
        rng.shuffle(fs)
        _pools.append(fs)
    pairings = list(zip(*_pools))

    if verbose:
        print(f"  [Combine] {protein_id}: {[len(fs) for fs in tail_files]} tail conformers -> "
              f"{len(pairings)} unique pairings (no reuse); graft both IDRs {idr_ranges} onto "
              f"fold {flo}-{fhi} + relax + validate.")

    done, attempts = 0, 0

    def _attempt(pairing):
        """Graft, relax, and validate one per-tail conformer pairing."""
        nonlocal done, attempts
        attempts += 1
        idr_specs = [dict(conf_path=pairing[j],
                          idr_range=tail_specs[j]["idr_range"], offset=tail_specs[j]["offset"])
                     for j in range(len(tail_specs))]
        grafted = os.path.join(work_dir, f"_grafted_{attempts}.pdb")
        reps = graft_back.graft_idrs_onto_fold(static_atoms, fold_range, idr_specs, grafted)
        if any(r.get("rms") is None for r in reps):
            if verbose:
                bad = [r["idr_range"] for r in reps if r.get("rms") is None]
                print(f"     [Attempt {attempts}] tail(s) {bad} failed stub alignment; skipping.")
            if os.path.exists(grafted):
                os.remove(grafted)
            return
        raw = load_pdb_structure(grafted, pdb_parser, verbose=verbose)
        if os.path.exists(grafted):
            os.remove(grafted)
        if not raw:
            return
        out_path = os.path.join(work_dir, f"stitched_conformer_{done + 1}.pdb")
        if verbose:
            print(f"\n{'-'*60}\n     [Attempt {attempts}] Combined chimera -> full length")
        try:
            is_valid, info = _relax_repair_validate(
                raw, out_path, region_resids, gate_ctx, expected_knot_type,
                attempts, done, pdb_parser, verbose=verbose, device=device)
        except Exception as e:
            if verbose:
                print(f"       [ERROR] Validation crashed: {e}")
            if os.path.exists(out_path):
                os.remove(out_path)
            return
        if is_valid:
            done += 1
            finalize_model_pdb(out_path, done)
            if verbose:
                print(f"       [RESULT] SUCCESS! (Total: {done}/{num_conformers})")
        elif verbose:
            print(f"       [RESULT] FAILED ({info.get('reason', 'Unknown')}) "
                  f"[Thresh: {info.get('_threshold', 0):.2f}]")

    # Phase 1: single pass
    for pairing in pairings:
        if done >= num_conformers:
            break
        _attempt(pairing)

    # Phase 2: top-up
    max_attempts = getattr(cfg, 'STITCH_MAX_ATTEMPTS', num_conformers * 5)
    if done < num_conformers:
        if verbose:
            print(f"    [Top-up] no-reuse pass gave {done}/{num_conformers}; filling the rest by "
                  f"random sampling WITH replacement (cap {max_attempts} attempts).", flush=True)
        while done < num_conformers and attempts < max_attempts:
            _attempt(tuple(rng.choice(tail_files[j]) for j in range(len(tail_specs))))
        if verbose and done < num_conformers:
            print(f"    [Note] reached {done}/{num_conformers} after {attempts} attempts "
                  f"(hit the {max_attempts}-attempt cap).", flush=True)

    accepted = sorted(glob.glob(os.path.join(work_dir, "stitched_conformer_*.pdb")),
                      key=lambda p: int(re.search(r'(\d+)\.pdb', p).group(1)))
    if not accepted:
        if verbose:
            print(f"    [!] No combined chimeras survived ({attempts} attempts); no ensemble written.")
        return "Failed", 0
    os.makedirs(final_dest_dir, exist_ok=True)
    _clear_stale_ensembles(final_dest_dir, protein_id)
    ensemble_path = os.path.join(final_dest_dir, f"{protein_id}_ensemble_n{len(accepted)}.pdb")
    with open(ensemble_path, 'w') as f:
        f.writelines(mkensemble(accepted))
    if verbose:
        print(f"    -> Combined + validated ensemble: {ensemble_path} "
              f"({len(accepted)} models, {attempts} attempts)")
    return get_completion_status(len(accepted), num_conformers), len(accepted)

# --- Main Processing Function ---
def process_protein(protein_id, labeled_db, id_to_pdb_path, conformer_root_dir, output_dir, final_output_root, num_conformers, length_ref=None, verbose=False, knot_screening=None, **kwargs):
    """Orchestrate the stitching/assembly pipeline for a single protein."""

    labeled_idrs = labeled_db[protein_id].get('labeled_idrs', [])

    # Identify IDR segments
    ldr_infos = [i for i in labeled_idrs if i.get('type') != 'IDP']
    category = get_protein_category(labeled_idrs)

    if not ldr_infos and category != "Category_0_IDP":
        return category, "Skipped", 0

    # Determine length label
    num_residues = length_ref.get(protein_id, 0) if length_ref else 0
    length_label = get_length_label(num_residues)

    # Determine directory paths
    mode = "minimized" if (len(ldr_infos) <= 1 or category == "Category_0_IDP") else "stitched"
    final_dest_dir = os.path.join(final_output_root, category, length_label, protein_id)
    work_dir = os.path.join(output_dir, category, protein_id, f"{mode}_ensemble")

    # Fast-pass: collect Step 3 conformers into one ensemble
    if category == "Category_0_IDP" or len(ldr_infos) <= 1:
        _have = _max_ensemble_n(final_dest_dir, protein_id)
        if _have >= num_conformers:
            if verbose:
                print(f"  [Resume] Ensemble already at n{_have} (>= {num_conformers}). Skipping.")
            return category, "Complete", _have

        if verbose:
            print(f"  [Fast-Pass] Single-region detected. Combining into ensemble PDB...")

        # Identify source folder
        if category == "Category_0_IDP":
            src_folder = os.path.join(conformer_root_dir, protein_id)
            source_files = []
            for root, dirs, files_in_dir in os.walk(src_folder):
                source_files.extend(
                    os.path.join(root, f) for f in files_in_dir if f.endswith("_validated.pdb")
                )
        else:
            idr_info = ldr_infos[0]
            start, end = idr_info['range']
            range_tag = f"idr_{start}-{end}"
            src_folder = os.path.join(conformer_root_dir, protein_id, range_tag)
            graft_spec = _load_graft_spec(src_folder)
            if graft_spec is not None:
                gs_static = id_to_pdb_path.get(protein_id)
                if gs_static:
                    _have = _max_ensemble_n(final_dest_dir, protein_id)
                    if _have >= num_conformers:
                        if verbose:
                            print(f"  [Resume] Ensemble already at n{_have} (>= {num_conformers}). Skipping.")
                        return category, "Complete", _have
                    gs_status, gs_n = reconstruct_truncated_pool(
                        protein_id, src_folder, graft_spec, gs_static, num_conformers,
                        work_dir, final_dest_dir, knot_screening=knot_screening, verbose=verbose)
                    return category, gs_status, gs_n
                elif verbose:
                    print(f"    [!] Truncated pool but no static PDB for {protein_id}; bundling as-is.")
            fp_path, fp_files = _maybe_renumber_truncated_pool(
                src_folder, work_dir, idr_info.get('label', 'D1'), verbose=verbose)
            if fp_files is not None:
                source_files = fp_files
            else:
                source_files = glob.glob(os.path.join(src_folder, "*_validated.pdb"))

        if not source_files:
            if verbose:
                print(f"    [!] Error: No validated conformers found in {src_folder}")
            return category, "Failed", 0

        # Order by conformer index
        def _conf_idx(p):
            m = re.search(r'(\d+)_validated', os.path.basename(p))
            return (0, int(m.group(1))) if m else (1, os.path.basename(p))
        source_files = sorted(source_files, key=_conf_idx)[:num_conformers]

        os.makedirs(final_dest_dir, exist_ok=True)
        _clear_stale_ensembles(final_dest_dir, protein_id)

        # Build ensemble PDB
        n_models = len(source_files)
        ensemble_filename = f"{protein_id}_ensemble_n{n_models}.pdb"
        ensemble_path = os.path.join(final_dest_dir, ensemble_filename)

        with open(ensemble_path, 'w') as f:
            f.writelines(mkensemble(source_files))

        if verbose:
            print(f"    -> Ensemble PDB saved: {ensemble_path} ({n_models} models)")
        return category, "Complete", n_models
    # --- Case 2: load static PDB ---
    static_path = id_to_pdb_path.get(protein_id)
    if not static_path:
        if verbose:
            print(f"    [!] Error: Static PDB for {protein_id} not found.")
        return category, "Failed", 0

    pdb_parser = PDBParser(QUIET=True)
    static_struct = load_pdb_structure(static_path, pdb_parser, verbose=verbose)
    if not static_struct: return category, "Failed", 0

    # --- RESUME LOGIC ---
    # 1. Check Final Destination for existing ensemble
    existing_final = 0
    if os.path.exists(final_dest_dir):
        _have = _max_ensemble_n(final_dest_dir, protein_id)
        if _have >= num_conformers:
            if verbose:
                print(f"  [Resume] Ensemble already at n{_have} (>= {num_conformers}). Skipping.")
            return category, "Complete", _have
        existing_final = len(glob.glob(os.path.join(final_dest_dir, f"{mode}_conformer_*.pdb")))

    # 2. Check Temp Work Dir
    max_temp_idx = 0
    if os.path.exists(work_dir):
        temp_files = glob.glob(os.path.join(work_dir, f"{mode}_conformer_*.pdb"))
        for f in temp_files:
            try:
                n = int(re.search(r"(\d+)\.pdb$", f).group(1))
                max_temp_idx = max(max_temp_idx, n)
            except: pass

    # 3. Determine 'Done' count
    done = max(existing_final, max_temp_idx)

    # Setup Source
    ensemble_dirs = find_ensemble_dirs(protein_id, conformer_root_dir, ldr_infos, verbose=verbose)
    if not ensemble_dirs and category != "Category_0_IDP":
        return category, "Failed", 0

    os.makedirs(work_dir, exist_ok=True)

    # --- Truncated multi-tail graft ---
    if ensemble_dirs and len(ldr_infos) >= 2:
        _pool_specs, _ok = [], True
        for _idr in ldr_infos:
            _lo, _hi = _idr['range']
            _match = None
            for _entry in ensemble_dirs.values():
                _s = _load_graft_spec(_entry['path'])
                if _s and any(tuple(r) == (_lo, _hi) for r in _s['idr_ranges']):
                    _match = (_entry['path'], _s)
                    break
            if _match is None:
                _ok = False
                break
            _pool_specs.append((_match[0], _match[1], (_lo, _hi)))
        if _ok:
            _fold_range = (max(s['fold_range'][0] for _, s, _ in _pool_specs),
                           min(s['fold_range'][1] for _, s, _ in _pool_specs))
            if _fold_range[0] <= _fold_range[1]:
                if verbose:
                    print(f"  [Graft] {len(_pool_specs)} truncated tails -> shared fold "
                          f"{_fold_range[0]}-{_fold_range[1]} (reconstruct_combined_chimera).")
                _tail_specs = [dict(pool_dir=_pd, idr_range=_ir, offset=_s['offset'])
                               for _pd, _s, _ir in _pool_specs]
                _st, _n = reconstruct_combined_chimera(
                    protein_id, _tail_specs, static_path, _fold_range, num_conformers,
                    work_dir, final_dest_dir, knot_screening=knot_screening, verbose=verbose,
                    device=kwargs.get('device', 'cuda:0'))
                return category, _st, _n

    # Map truncated sub-ensembles back to full-length numbering
    if ensemble_dirs:
        for _label, _entry in ensemble_dirs.items():
            _new_path, _new_files = _maybe_renumber_truncated_pool(
                _entry['path'], work_dir, _label, verbose=verbose)
            if _new_files is not None:
                _entry['path'] = _new_path
                _entry['files'] = _new_files

    region_resids = build_region_resids(ldr_infos)
    _HIS_RESNAMES = {'HIS', 'HID', 'HIE', 'HIP'}

    # --- Folded-domain curvature gate setup ---
    curv_ratio = kwargs.get('fold_curv_ratio', getattr(cfg, 'STITCH_FOLD_CURV_RATIO', 0.0)) or 0.0
    curv_window = int(kwargs.get('fold_curv_window', getattr(cfg, 'STITCH_FOLD_CURV_WINDOW', 15)))
    gate_ref_ca = gate_folded_mask = None
    gate_n_res = 0
    if curv_ratio > 0:
        gate_n_res = len([r for r in static_struct[0].get_list()[0] if r.id[0] == ' '])
        gate_ref_ca = _ca_from_pdb_file(static_path, gate_n_res)
        gate_folded_mask = np.ones(gate_n_res, dtype=bool)
        for _idr in ldr_infos:
            _lo, _hi = _idr['range']
            gate_folded_mask[_lo - 1:_hi] = False

    # Expected knot type
    expected_knot_type = None
    if knot_screening:
        expected_knot_type = knot_screening.get(protein_id)

    # Loop
    if verbose:
        if expected_knot_type is not None:
            print(f"  [Topology] Native spec: {format_domain_spec(expected_knot_type)} "
                  f"(each folded domain's knot must match)")
        print(f"  Generating {num_conformers} conformers ({mode})...")
        print(f"  [Resume] Found {existing_final} in final, {max_temp_idx} in temp. Starting at {done+1}.")

    attempts = 0

    while done < num_conformers and attempts < cfg.STITCH_MAX_ATTEMPTS:
        attempts += 1

        if verbose and attempts % 50 == 0:
            print(f"     [PROGRESS] Summary: {done}/{num_conformers} successes")

        if verbose:
            print("\n" + "-"*60)

        out_name = f"{mode}_conformer_{done+1}.pdb"
        out_path = os.path.join(work_dir, out_name)

        # --- GENERATION: Assemble Kinematic Chain ---
        if verbose:
            print(f"     [Attempt {attempts}] Assembling Kinematic Chain...")
        raw_s = assemble_kinematic_chain(
            static_struct, ensemble_dirs, ldr_infos,
            set(r.id[1] for r in static_struct[0].get_list()[0] if r.id[0] == ' '),
            pdb_parser
        )
        raw = clean_structure(raw_s) if raw_s else None

        if not raw: continue

        # --- Shared relax + validation ---
        gate_ctx = ({"ref_ca": gate_ref_ca, "folded_mask": gate_folded_mask,
                     "n_res": gate_n_res, "curv_ratio": curv_ratio, "curv_window": curv_window}
                    if (curv_ratio > 0 and gate_ref_ca is not None) else None)
        try:
            is_valid, info = _relax_repair_validate(
                raw, out_path, region_resids, gate_ctx, expected_knot_type,
                attempts, done, pdb_parser, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"       [ERROR] Validation crashed: {e}")
            if os.path.exists(out_path): os.remove(out_path)
            continue

        if is_valid:
            done += 1
            finalize_model_pdb(out_path, done)
            if verbose:
                print(f"       [RESULT] SUCCESS! (Total: {done}/{num_conformers})")
        elif verbose:
            print(f"       [RESULT] FAILED ({info.get('reason', 'Unknown')}) "
                  f"[Thresh: {info.get('_threshold', 0):.2f}]")

    status = get_completion_status(done, num_conformers)

    # --- Combine conformers into one multi-model PDB ---
    conformer_files = sorted(
        glob.glob(os.path.join(work_dir, f"{mode}_conformer_*.pdb")),
        key=lambda p: int(re.search(r'(\d+)\.pdb$', p).group(1))
    )

    if conformer_files:
        n_models = len(conformer_files)
        os.makedirs(final_dest_dir, exist_ok=True)
        _clear_stale_ensembles(final_dest_dir, protein_id)
        ensemble_filename = f"{protein_id}_ensemble_n{n_models}.pdb"
        ensemble_path = os.path.join(final_dest_dir, ensemble_filename)

        with open(ensemble_path, 'w') as f:
            f.writelines(mkensemble(conformer_files))

        if verbose:
            print(f"    -> Ensemble PDB saved: {ensemble_path} ({n_models} models)")

    # Cleanup temp working directory
    shutil.rmtree(work_dir, ignore_errors=True)
    try:
        os.rmdir(os.path.dirname(work_dir))
    except OSError:
        pass

    return category, status, done

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4: Kinematic Stitching & Energy Minimization Pipeline",
        epilog="Example: python Step_4_stitch.py --id_file ids.txt --total_splits 10 --split_index 0"
    )
    parser.add_argument("--id_file", default=cfg.RUN_ID_FILE,
                        help=f"Newline-separated UniProt ID list (default from config: {cfg.RUN_ID_FILE}).")
    parser.add_argument("--total_splits", type=int, default=1,
                        help="Total number of parallel jobs (default: 1).")
    parser.add_argument("--split_index", type=int, default=0,
                        help="The specific shard index, 0-based (default: 0).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for local execution (default: 1, sequential).")
    parser.add_argument("--labeled_db", default=cfg.LABELED_DB_PATH,
                        help=f"Path to labeled database JSON (default: {cfg.LABELED_DB_PATH}).")
    parser.add_argument("--length_ref", default=cfg.LENGTH_REF_PATH,
                        help=f"Path to residue-count reference JSON (default: {cfg.LENGTH_REF_PATH}).")
    parser.add_argument("--conformer_dir", default=cfg.CONFORMER_POOL_DIR,
                        help=f"Directory containing Step 3 conformer pools (default: {cfg.CONFORMER_POOL_DIR}).")
    parser.add_argument("--output_dir", default=cfg.STITCH_OUTPUT_ROOT,
                        help=f"Root output directory for final models (default: {cfg.STITCH_OUTPUT_ROOT}).")
    parser.add_argument("--n_conformers", type=int, default=cfg.STITCH_N_CONFORMERS,
                        help=f"Target number of ensemble conformers per protein (default: {cfg.STITCH_N_CONFORMERS}).")
    parser.add_argument("--max_attempts", type=int, default=cfg.STITCH_MAX_ATTEMPTS,
                        help=f"Maximum stitching attempts per protein (default: {cfg.STITCH_MAX_ATTEMPTS}).")
    parser.add_argument("--fold_curv_ratio", type=float, default=cfg.STITCH_FOLD_CURV_RATIO,
                        help=f"Folded-domain curvature gate: reject if a junction-adjacent fold window "
                             f"straightens below this fraction of the template curvature; 0 disables "
                             f"(default: {cfg.STITCH_FOLD_CURV_RATIO}).")
    parser.add_argument("--fold_curv_window", type=int, default=cfg.STITCH_FOLD_CURV_WINDOW,
                        help=f"Folded residues into the fold from each junction to average for the "
                             f"curvature gate (default: {cfg.STITCH_FOLD_CURV_WINDOW}).")
    parser.add_argument("--verbose", action="store_true", default=cfg.VERBOSE,
                        help="Enable detailed per-protein logging.")
    args = parser.parse_args()

    # Override config with CLI args
    cfg.STITCH_MAX_ATTEMPTS = args.max_attempts
    cfg.STITCH_FOLD_CURV_RATIO = args.fold_curv_ratio
    cfg.STITCH_FOLD_CURV_WINDOW = args.fold_curv_window

    batch_label = os.path.splitext(os.path.basename(args.id_file))[0]

    print(f"\n--- Initializing Workflow for: {batch_label} ---")

    try:
        with open(args.id_file, 'r') as f:
            protein_ids_to_run = sorted({line.strip() for line in f if line.strip()})
    except FileNotFoundError:
        print(f"[!] Error: ID file not found at {args.id_file}")
        sys.exit(1)

    total_ids = len(protein_ids_to_run)
    if args.total_splits > 1:
        chunks = np.array_split(protein_ids_to_run, args.total_splits)
        if 0 <= args.split_index < len(chunks):
            chunk_of_ids = chunks[args.split_index].tolist()
        else:
            print(f"[!] Warning: Split index {args.split_index} out of bounds. Processing empty list.")
            chunk_of_ids = []
        print(f"    Mode:        PARALLEL (Split {args.split_index + 1} of {args.total_splits})")
        print(f"    Target Load: {len(chunk_of_ids)} proteins (out of {total_ids} total)")
    else:
        chunk_of_ids = protein_ids_to_run
        print(f"    Mode:        SINGLE THREAD")
        print(f"    Target Load: {total_ids} proteins")

    try:
        id_to_pdb_path = get_id_to_pdb_path()
        with open(args.labeled_db, 'r') as f:
            labeled_db = json.load(f)
        with open(args.length_ref, 'r') as f:
            length_ref = json.load(f)
        print(f"Configuration and Databases loaded.")
    except Exception as e:
        print(f"[!] CRITICAL ERROR: Failed to load external resources: {e}")
        sys.exit(1)

    # Load knot screening
    knot_screening_path = getattr(cfg, 'KNOT_SCREENING_PATH',
        os.path.join(cfg.INPUT_DATA_DIR, "knot_screening.json"))
    knot_screening = load_knot_screening(knot_screening_path)
    if knot_screening:
        n_knotted = sum(
            1 for spec in knot_screening.values()
            if any(d.get("knot") for d in spec)
        )
        print(f"    Knot screening: {len(knot_screening)} entries "
              f"({n_knotted} with at least one natively knotted domain).")
    else:
        print(f"    Knot screening: not available (full-chain unknot baseline will be applied).")

    final_root_dir = args.output_dir
    temp_working_dir = os.path.join(final_root_dir, f"_temp_work_{batch_label}_{args.split_index}")

    os.makedirs(final_root_dir, exist_ok=True)
    os.makedirs(temp_working_dir, exist_ok=True)

    print(f"    Output Root: {final_root_dir}")
    print(f"    Temp Work:   {temp_working_dir}")

    verbose = args.verbose
    if verbose:
        print(f"\n--- Starting Processing Loop ---")

    stats = Counter()
    status_counts = Counter()
    total_in_chunk = len(chunk_of_ids)

    # Common kwargs for process_protein
    common_kwargs = dict(
        labeled_db=labeled_db,
        id_to_pdb_path=id_to_pdb_path,
        conformer_root_dir=args.conformer_dir,
        output_dir=temp_working_dir,
        final_output_root=final_root_dir,
        num_conformers=args.n_conformers,
        length_ref=length_ref,
        verbose=verbose,
        knot_screening=knot_screening,
        fold_curv_ratio=args.fold_curv_ratio,
        fold_curv_window=args.fold_curv_window
    )

    # Filter to valid IDs upfront
    valid_ids = []
    for protein_id in chunk_of_ids:
        if protein_id not in labeled_db:
            if verbose:
                print(f"    -> SKIPPED: {protein_id} not found in labeled DB.")
            stats['skipped'] += 1
        else:
            valid_ids.append(protein_id)

    if args.workers > 1:
        # --- PARALLEL EXECUTION ---
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"    Workers:     {args.workers}")
        perprotein_dir = os.path.join("logs", "Step4", "perprotein", f"split_{args.split_index}")
        os.makedirs(perprotein_dir, exist_ok=True)
        print(f"    Per-protein LIVE logs: {perprotein_dir}/<ID>.log   (tail -f to watch one)")
        print(f"    {'#':>4s}  {'protein':14s} {'status':10s} {'models':>6s}   log")

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_protein_to_file, protein_id=pid,
                                _log_dir=perprotein_dir, **common_kwargs): pid
                for pid in valid_ids
            }

            done_n = 0
            for fut in as_completed(futures):
                protein_id = futures[fut]
                done_n += 1
                tag = f"[{done_n}/{len(valid_ids)}]"
                try:
                    result, log_path, err = fut.result()
                except Exception as e:
                    print(f"    {tag:>7s}  {protein_id:14s} {'WORKER-ERR':10s} {'':>6s}   {e}", flush=True)
                    stats['crashed'] += 1
                    continue
                logname = os.path.basename(log_path)
                if err is not None:
                    print(f"    {tag:>7s}  {protein_id:14s} {'CRASH':10s} {'':>6s}   {logname}", flush=True)
                    stats['crashed'] += 1
                    continue
                cat_result, completion_status, num_success = result
                stats['processed'] += 1
                status_counts[completion_status] += 1
                if completion_status == "Failed": stats['failed'] += 1
                print(f"    {tag:>7s}  {protein_id:14s} {completion_status:10s} {num_success:>6d}   {logname}", flush=True)
    else:
        # --- SEQUENTIAL EXECUTION ---
        iterator = valid_ids

        for i, protein_id in enumerate(iterator, 1):
            if verbose:
                print(f"\n--- [{i}/{len(valid_ids)}] Processing Protein: {protein_id} ---")

            try:
                cat_result, completion_status, num_success = process_protein(
                    protein_id=protein_id, **common_kwargs
                )

                stats['processed'] += 1
                status_counts[completion_status] += 1

                if verbose:
                    symbol = "\u2713" if completion_status == "Complete" else "!"
                    print(f"    -> {symbol} Result: {completion_status} ({num_success} models)")
                if completion_status == "Failed": stats['failed'] += 1
            except Exception as e:
                if verbose:
                    print(f"    [!] EXCEPTION CRASH on {protein_id}: {e}")
                stats['crashed'] += 1

    try:
        shutil.rmtree(temp_working_dir, ignore_errors=True)
    except OSError as e:
        print(f"[!] Warning: Could not clean up temp dir: {e}")

    print(f"\n" + "="*40)
    print(f"   STEP 4 BATCH REPORT: {batch_label}")
    print(f"   Split {args.split_index + 1}/{args.total_splits}")
    print(f"="*40)
    print(f" Total Proteins : {total_in_chunk}")
    print(f" Processed      : {stats['processed']}")
    print(f" Skipped (No DB): {stats['skipped']}")
    print(f" Crashed        : {stats['crashed']}")
    print(f"-"*40)
    print(f" Status Breakdown:")
    for status, count in status_counts.items():
        print(f"   - {status:<18}: {count}")
    print(f"="*40 + "\n")