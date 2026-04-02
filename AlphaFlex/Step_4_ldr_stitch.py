"""
Step 4: Stitching & Relaxation Pipeline
===============================

This script orchestrates the high-throughput assembly of full-length protein models 
by stitching pre-generated disordered ensembles onto folded domains. It is designed 
for scalability, supporting parallel execution across High-Performance Computing (HPC) 
clusters via deterministic list sharding.

Usage:
    # Single Process (Default)
    python Step_4_multiple_ldr_stitch.py --id_file ids.txt

    # Parallel Execution (e.g., Job 1 of 10)
    python Step_4_multiple_ldr_stitch.py --id_file ids.txt --total_splits 10 --split_index 0

Workflow:
    1. Input Handling: Reads protein IDs and performs deterministic sorting to ensure 
       consistent sharding across parallel jobs.
    2. Splitting (Optional): If --total_splits > 1, mathematically divides the sorted 
       list into N equal chunks and selects the subset corresponding to --split_index.
    3. Resource Mapping: Maps IDs to AlphaFold2 anchors and IDP conformer pools.
    4. Execution: Runs the Monte Carlo Stitching -> Relaxation -> Validation pipeline 
       for each protein in the assigned chunk.
    5. Output: Writes results to a unique temporary directory (to prevent race conditions) 
       before finalizing them in the shared output root.

Arguments:
    --id_file (str):      
        Path to a text file containing newline-separated UniProt IDs to process.
    
    --total_splits (int, default=1): 
        Total number of parallel jobs (shards) the input list is divided into. 
        Set this to the size of your SLURM job array.
    
    --split_index (int, default=0):  
        The specific shard index (0-based) to process in this execution instance.
        Must be strictly less than --total_splits.
"""
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
from pdbtools import pdb_mkensemble


def mkensemble(pdb_files):
    """Wrapper around pdb_mkensemble.run that ensures each line ends with \\n.
    pdb-tools' pad_line can swallow the newline on TER records, causing
    ENDMDL to be appended to the TER line instead of its own line."""
    for line in pdb_mkensemble.run(pdb_files):
        yield line.rstrip('\n') + '\n'


# --- BioPython Setup ---
try:
    from Bio.PDB import PDBParser, PDBIO
    print("BioPython.PDB loaded successfully.")
except ImportError:
    print("Error: BioPython is required. Run: conda install -c conda-forge biopython")
    sys.exit(1)
# ---

# --- OpenMM Setup ---
try:
    from openmm.app import PDBFile
    print("OpenMM loaded successfully.")
except ImportError:
    print("Error: OpenMM is required for this script.")
    print("Please install it in your environment: conda install -c conda-forge openmm")
    sys.exit(1)
# ---

# --- Relax.py and Config Import Setup ---
from idpforge.utils.relax import relax_protein
# ---

# --- Import Config ---
try:
    import config as cfg
except ImportError:
    print("Error: auto_config.py not found. Please create it.")
    sys.exit(1)
# ---

# --- Shared Utility Imports (same as Step 3) ---
from utils.smart_scoring import get_smart_threshold
from idpforge.utils.structure_repair import repair_chirality, fix_histidine_naming
from idpforge.utils.structure_validation import validate_structure_post_relax, check_bond_integrity
# ---

# --- Stitch Utility Imports ---
from utils.stitch import (
    get_completion_status, get_length_label, build_region_resids,
    get_id_to_pdb_path, get_protein_category, find_ensemble_dirs,
    format_ranges, load_pdb_structure, get_segment_atoms,
    build_segment_map, assemble_kinematic_chain, clean_structure
)
# ---


# --- Energy Minimization with Amber ---
relax_cfg = {
    'max_outer_iterations': cfg.RELAX_MAX_OUTER_ITER,
    'stiffness': cfg.RELAX_STIFFNESS,
    'exclude_residues': [],
    'max_iterations': cfg.MINIMIZATION_MAX_ITER,
    'tolerance': cfg.MINIMIZATION_TOLERANCE
}

def relax_with_established_method(structure, output_filepath, idr_indices=None, device="cuda:0", verbose=False):
    """
    Performs energy minimization using the AMBER99SB forcefield via OpenMM.

    The folded domains are harmonically restrained to preserve their predicted structure,
    while the stitched junctions and IDRs are allowed to relax, resolving local
    steric clashes and bond length distortions.
    """
    pdb_name = os.path.splitext(os.path.basename(output_filepath))[0]
    output_dir = os.path.dirname(output_filepath)

    # 1. Setup Config for this specific run
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
        result = relax_protein(
            config=run_config,
            model_device=device,
            unrelaxed_protein=unrelaxed_prot,
            output_dir=output_dir,
            pdb_name=pdb_name,
            viol_threshold=0.02
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
# ----------------------------------

# --- Main Processing Function ---
def process_protein(protein_id, labeled_db, id_to_pdb_path, conformer_root_dir, output_dir, final_output_root, num_conformers, length_ref=None, verbose=False, **kwargs):
    """
    Orchestrates the assembly pipeline for a single protein.

    Case 1 (Single IDR / IDP):
        Collects all validated conformers from Step 3 and writes them into a
        single multi-model PDB ensemble file.

    Case 2 (Multiple IDRs):
        Runs a Monte Carlo stitching loop:
        1. Stitch: Assembles a new conformation via kinematic chain assembly.
        2. Relax: AMBER energy minimization with folded-domain restraints.
        3. Repair: Chirality + HIS naming repair, re-relax if needed.
        4. Validate: Unified validation (chirality, bonds, clashes, topology).
        5. Combine: Writes all valid conformers into one multi-model PDB.

    Returns:
        tuple: (Category, Status, Success_Count) for global reporting.
    """

    labeled_idrs = labeled_db[protein_id].get('labeled_idrs', [])

    # Identify specific IDR segments (excluding the global IDP label)
    ldr_infos = [i for i in labeled_idrs if i.get('type') != 'IDP']
    category = get_protein_category(labeled_idrs)

    # Skip if no disordered regions found
    if not ldr_infos and category != "Category_0_IDP":
        return category, "Skipped", 0

    # Determine length label from pre-loaded reference (avoids parsing static PDB)
    num_residues = length_ref.get(protein_id, 0) if length_ref else 0
    length_label = get_length_label(num_residues)

    # Determine directory paths
    mode = "minimized" if (len(ldr_infos) <= 1 or category == "Category_0_IDP") else "stitched"
    final_dest_dir = os.path.join(final_output_root, category, length_label, protein_id)
    work_dir = os.path.join(output_dir, category, protein_id, f"{mode}_ensemble")

    # =========================================================================
    # FAST-PASS LOGIC: Collect all Step 3 conformers into a single ensemble PDB
    # =========================================================================
    if category == "Category_0_IDP" or len(ldr_infos) <= 1:
        # Skip if ensemble already exists in final destination
        if os.path.exists(final_dest_dir):
            existing = glob.glob(os.path.join(final_dest_dir, f"{protein_id}_ensemble_*.pdb"))
            if existing:
                if verbose:
                    print(f"  [Resume] Ensemble PDB already exists. Skipping.")
                return category, "Complete", num_conformers

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
            source_files = glob.glob(os.path.join(src_folder, "*_validated.pdb"))

        if not source_files:
            if verbose:
                print(f"    [!] Error: No validated conformers found in {src_folder}")
            return category, "Failed", 0

        source_files = sorted(source_files)[:num_conformers]

        os.makedirs(final_dest_dir, exist_ok=True)

        # Build multi-model ensemble PDB using pdb-tools
        n_models = len(source_files)
        ensemble_filename = f"{protein_id}_ensemble_n{n_models}.pdb"
        ensemble_path = os.path.join(final_dest_dir, ensemble_filename)

        with open(ensemble_path, 'w') as f:
            f.writelines(mkensemble(source_files))

        if verbose:
            print(f"    -> Ensemble PDB saved: {ensemble_path} ({n_models} models)")
        return category, "Complete", n_models
    # =========================================================================
    # --- CASE 2: Load static PDB (only needed for stitching) ---
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
        existing_ensemble = glob.glob(os.path.join(final_dest_dir, f"{protein_id}_ensemble_*.pdb"))
        if existing_ensemble:
            if verbose:
                print(f"  [Resume] Ensemble PDB already exists. Skipping.")
            return category, "Complete", num_conformers
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
    # ----------------------------

    # Setup Source
    ensemble_dirs = find_ensemble_dirs(protein_id, conformer_root_dir, ldr_infos, verbose=verbose)
    if not ensemble_dirs and category != "Category_0_IDP":
        return category, "Failed", 0

    os.makedirs(work_dir, exist_ok=True)

    region_resids = build_region_resids(ldr_infos)
    _HIS_RESNAMES = {'HIS', 'HID', 'HIE', 'HIP'}

    # Loop
    if verbose:
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

        # Identify Frozen/IDR Residues
        idr_idx, frozen_ids, cnt = [], [], 0

        all_residues_in_struct = list(raw[0].get_residues())
        all_residues_in_struct.sort(key=lambda x: x.id[1])

        for r in all_residues_in_struct:
            if r.id[0] == ' ':
                if r.id[1] in region_resids:
                    idr_idx.append(cnt)
                else:
                    frozen_ids.append(r.id[1])
                cnt += 1

        if verbose:
            print(f"       [CONFIG] Freezing residues: {format_ranges(frozen_ids)}")

        # --- PRE-MINIMIZATION: Save and repair chirality ---
        io_save = PDBIO()
        io_save.set_structure(raw)
        io_save.save(out_path)
        repair_chirality(out_path, verbose=False)

        repaired = load_pdb_structure(out_path, pdb_parser, verbose=verbose) or raw

        # --- RELAXATION ---
        if not relax_with_established_method(repaired, out_path, idr_indices=idr_idx, verbose=verbose):
            if verbose:
                print(f"       [RESULT] FAILED (Relaxation Rejected)")
            if os.path.exists(out_path): os.remove(out_path)
            continue

        # --- POST-RELAX REPAIR (matching Step 3 pattern) ---
        needs_rerelax = False

        # Bond integrity check for broken HIS residues
        try:
            chk_pdb = PDBFile(out_path)
            broken = check_bond_integrity(chk_pdb.topology, chk_pdb.positions)
            his_resids = {b['resid'] for b in broken
                          if b['resname'] in _HIS_RESNAMES
                          or b.get('resname2', '') in _HIS_RESNAMES}
        except Exception:
            his_resids = set()

        # Chirality repair (post-relax)
        n_chiral = repair_chirality(out_path, verbose=False)
        if n_chiral > 0:
            needs_rerelax = True
            if verbose:
                print(f"       [REPAIR] Flipped {n_chiral} D-isomer(s).")

        # HIS naming fix
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

        # Re-relax if repairs were applied
        if needs_rerelax:
            repaired_struct = load_pdb_structure(out_path, pdb_parser, verbose=verbose)
            if not repaired_struct or not relax_with_established_method(
                repaired_struct, out_path, idr_indices=idr_idx, verbose=verbose
            ):
                if verbose:
                    print(f"       [RESULT] FAILED (Re-relax after repair)")
                if os.path.exists(out_path): os.remove(out_path)
                continue

        # --- POST-MINIMIZATION VALIDATION (shared utils) ---
        if verbose:
            print(f"       [POST-MIN CHECK] Validating...")
        try:
            chk = PDBFile(out_path)

            threshold = get_smart_threshold(
                attempts, done,
                base=cfg.STITCH_BASE_CLASH_THRESHOLD,
                inc=cfg.STITCH_CLASH_INCREMENT
            )

            idr_start_res = min(region_resids) if region_resids else None
            idr_end_res = max(region_resids) if region_resids else None

            is_valid, info = validate_structure_post_relax(
                chk.topology, chk.positions,
                pdb_path=out_path,
                strict_clash_threshold=threshold,
                idr_start=idr_start_res,
                idr_end=idr_end_res,
                verbose=False,
                full_report=True
            )

            if verbose:
                # Log results (matching Step 3 format)
                chiral_str = "PASS" if info.get("chirality_pass", True) else "FAIL"
                bonds_str = "PASS" if info.get("bonds_pass", True) else f"FAIL ({info.get('num_broken_bonds', 0)} broken)"
                clash_str = "PASS" if info.get("clash_pass", True) else "FAIL"
                knot_str = "PASS" if info.get("knot_pass", True) else f"FAIL ({info.get('knot_type')})"

                print(f"         - Chirality: {chiral_str}")
                print(f"         - Bonds:     {bonds_str}")
                print(f"         - Clashes:   {info.get('num_clashes', '?')} (Score: {info.get('clash_score', 0):.2f} | Limit: {threshold:.1f}) -> {clash_str}")
                print(f"         - Topology:  {knot_str}")

            if is_valid:
                done += 1
                if verbose:
                    print(f"       [RESULT] SUCCESS! (Total: {done}/{num_conformers})")
            else:
                reason = info.get('reason', 'Unknown')
                if verbose:
                    print(f"       [RESULT] FAILED ({reason}) [Thresh: {threshold:.2f}]")
                if os.path.exists(out_path): os.remove(out_path)

        except Exception as e:
            if verbose:
                print(f"       [ERROR] Validation crashed: {e}")
            if os.path.exists(out_path): os.remove(out_path)

    status = get_completion_status(done, num_conformers)

    # --- Combine all conformers into a single multi-model PDB ---
    conformer_files = sorted(
        glob.glob(os.path.join(work_dir, f"{mode}_conformer_*.pdb")),
        key=lambda p: int(re.search(r'(\d+)\.pdb$', p).group(1))
    )

    if conformer_files:
        n_models = len(conformer_files)
        os.makedirs(final_dest_dir, exist_ok=True)
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
# ----------------------------------

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4: Kinematic Stitching & Energy Minimization Pipeline",
        epilog="Example: python Step_4_stitch.py --id_file ids.txt --total_splits 10 --split_index 0"
    )
    parser.add_argument("--id_file", required=True,
                        help="Path to the input text file containing newline-separated UniProt IDs.")
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
    parser.add_argument("--verbose", action="store_true", default=cfg.VERBOSE,
                        help="Enable detailed per-protein logging.")
    args = parser.parse_args()

    # Override config with CLI args so helper functions pick them up
    cfg.STITCH_MAX_ATTEMPTS = args.max_attempts

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
        verbose=verbose
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

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_protein, protein_id=pid, **common_kwargs): pid
                for pid in valid_ids
            }

            completed = as_completed(futures)

            for fut in completed:
                protein_id = futures[fut]
                try:
                    cat_result, completion_status, num_success = fut.result()
                    stats['processed'] += 1
                    status_counts[completion_status] += 1
                    if completion_status == "Failed": stats['failed'] += 1
                except Exception as e:
                    if verbose:
                        print(f"    [!] EXCEPTION CRASH on {protein_id}: {e}")
                    stats['crashed'] += 1
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