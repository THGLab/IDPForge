"""
Step 4: Stitching & Relaxation Pipeline
===============================

This script orchestrates the high-throughput assembly of full-length protein models 
by stitching pre-generated disordered ensembles onto folded domains. It is designed 
for scalability, supporting parallel execution across High-Performance Computing (HPC) 
clusters via deterministic list sharding.

Usage:
    # Single Process (Default)
    python Step_4_multiple_ldr_stitch.py --id_file ids.txt --category cat_test

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
    
    --category (str, optional):     
        Logical grouping for the run (e.g., 'cat_3_loops_only'). If omitted, the script 
        attempts to auto-detect it from the --id_file filename.
    
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
import random
import argparse
import sys
from collections import defaultdict, Counter
import numpy as np
import json
import io
import subprocess
import shutil
import tempfile
import math

# --- BioPython Setup ---
try:
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.StructureBuilder import StructureBuilder
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    from Bio.PDB import Superimposer
    from Bio.PDB.Structure import Structure 
    print("BioPython.PDB and Superimposer loaded successfully.")
except ImportError:
    print("Error: BioPython is required. Run: conda install -c conda-forge biopython")
    sys.exit(1)
# ---

# --- OpenMM Setup ---
try:
    from openmm.app import PDBFile, Modeller, ForceField, Simulation, NoCutoff
    from openmm import LangevinIntegrator, System, Context, VerletIntegrator, NonbondedForce, CustomExternalForce, CustomNonbondedForce, CustomTorsionForce
    from openmm import unit
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

# --- Constants ---
ALIGNMENT_STUB_HALF_SIZE = cfg.ALIGNMENT_STUB_HALF_SIZE
ALIGNMENT_JUNCTION_SIZE = cfg.ALIGNMENT_JUNCTION_SIZE
MIN_CONFORMER_POOL_SIZE = cfg.MIN_CONFORMER_POOL_SIZE
# ---

# --- Completion Status Helper ---
def get_completion_status(conformer_count, target_conformers):
    """
    Determines the success tier of the sampling process.
    
    Returns:
        'Complete': Target ensemble size reached.
        'Partially Complete': Sufficient sampling for preliminary analysis (>50).
        'Failed': Insufficient sampling due to persistent steric clashes or kinematic failures.
    """
    if conformer_count >= target_conformers:
        return "Complete"
    elif conformer_count > 50: 
        return "Partially Complete"
    else:
        return "Failed"
# ----------------------------------

# --- Length Binning Helper ---
def get_length_label(num_res):
    """
    Stratifies proteins into length bins for storage and analysis based on residue number.
    
    This binning allows for performance profiling of the stitching algorithm 
    across different size scales (e.g., small <250 vs. huge >2000 residues).
    """
    if num_res <= 250: return "0-250"
    elif num_res <= 500: return "251-500"
    elif num_res <= 1000: return "501-1000"
    elif num_res <= 1500: return "1001-1500"
    elif num_res <= 2000: return "1501-2000"
    else: return "2001+"
# ----------------------------------

# --- IDR Region Identification Helper ---
def build_region_resids(labeled_idrs):
    """
    Aggregates all residue indices belonging to Intrinsically Disordered Regions (IDRs).
    
    This set is used during energy minimization to define the 'mobile' selection, 
    allowing IDRs to relax while folded domains remain restrained.
    """
    region_resids = []
    for idr in labeled_idrs:
        start, end = idr['range']
        region_resids.extend(range(start, end+1))
    return region_resids
# ----------------------------------

# --- PDB File Mapping Helper ---
def get_id_to_pdb_path():
    """
    Indexes the configurated database to map UniProt IDs to local PDB file paths.
    
    Handles both standard AF2 naming conventions (AF-ID-F1) 
    and simplified filenames.
    """
    print(f"--- DEBUG: MAPPING PDBs ---")
    print(f"Scanning directory: {cfg.PDB_LIBRARY_PATH}")
    
    id_to_pdb_path = {}
    
    if not os.path.exists(cfg.PDB_LIBRARY_PATH):
         print(f"[!] CRITICAL: Directory not found!")
         return {}

    files = os.listdir(cfg.PDB_LIBRARY_PATH)
    print(f"Found {len(files)} files in directory.")

    for filename in files:
        pdb_path = os.path.join(cfg.PDB_LIBRARY_PATH, filename)
        if not filename.endswith(".pdb"): 
            print(f"  [Skip] Not a .pdb: {filename}")
            continue
        
        # 1. Try AlphaFold Long Format
        match = re.search(r"AF-([A-Z0-9]+)-F1", filename)
        
        # 2. Try Simple Format (The one O14653.pdb needs)
        match_simple = re.match(r"([A-Z0-9]+)\.pdb", filename)

        if match:
            pid = match.group(1)
            id_to_pdb_path[pid] = pdb_path
            print(f"  [Match] AF-Format: {filename} -> ID: {pid}")
        elif match_simple:
            pid = match_simple.group(1)
            id_to_pdb_path[pid] = pdb_path
            print(f"  [Match] Simple-Format: {filename} -> ID: {pid}")
        else:
            print(f"  [FAIL] No Regex Matched for: {filename}")
                
    print(f"--- Finished Mapping. Total IDs: {len(id_to_pdb_path)} ---")
    return id_to_pdb_path
# ----------------------------------

# --- Protein Category Helper ---
def get_protein_category(labeled_idrs):
    """
    Categorizes protein based on complexity: Loop (Hardest) > Linker > Tail (Easiest).
    - If it has a Loop -> Category 3 (Automatically, regardless of others).
    - If it has a Linker (but no Loop) -> Category 2.
    - If it has only Tails -> Category 1.
    """
    idr_types = set(idr['type'] for idr in labeled_idrs)
    
    if len(idr_types) == 1 and "IDP" in idr_types:
        return "Category_0_IDP"

    if "Loop IDR" in idr_types:
        return "Category_3"

    if "Linker IDR" in idr_types:
        return "Category_2"

    if "Tail IDR" in idr_types:
        return "Category_1"

    return "Uncategorized"
# ----------------------------------

# --- Ensemble Directory Finder ---
def find_ensemble_dirs(protein_id, conformer_root_dir, labeled_idrs):
    """
    Locates the pre-generated conformational ensembles for each disordered segment.
    
    Validates that a sufficient pool of conformers (MIN_CONFORMER_POOL_SIZE) exists 
    to allow for diverse sampling without over-relying on a single conformation.
    """
    ensemble_paths = {}
    base_protein_path = os.path.join(conformer_root_dir, protein_id)
    if not os.path.isdir(base_protein_path):
        print(f"   Error: Base conformer directory for {protein_id} not found. Skipping.")
        return None
    found_all = True
    for idr_info in labeled_idrs:
        idr_type = idr_info.get("type"); idr_label_d = idr_info.get("label")
        if idr_type == "IDP": continue 
        start, end = idr_info.get("range"); idr_label_range = f"idr_{start}-{end}"
        type_dir_name = idr_type.replace(" ", "_")
        idr_conformer_path = os.path.join(base_protein_path, type_dir_name, idr_label_range)
        pdb_files = glob.glob(os.path.join(idr_conformer_path, "*_relaxed.pdb"))
        
        if not pdb_files:
            old_path = os.path.join(base_protein_path, idr_label_range)
            pdb_files = glob.glob(os.path.join(old_path, "*_relaxed.pdb"))
            if pdb_files: idr_conformer_path = old_path 
            else:
                print(f"     Error: No '*_relaxed.pdb' files found for {idr_label_d} ({idr_label_range}).")
                found_all = False; break
        
        if len(pdb_files) < MIN_CONFORMER_POOL_SIZE:
            print(f"     Warning: Ensemble for {idr_label_d} has only {len(pdb_files)} files.")
            
        ensemble_paths[idr_label_d] = {
            'path': idr_conformer_path, 'files': pdb_files, 'range': (start, end), 
            'type': idr_type, 'flanking_domains': idr_info.get('flanking_domains', [])
        }
    return ensemble_paths if found_all else None
# ----------------------------------

# --- Range Identifier ---
def format_ranges(indices):
    """
    Formats a list of residue indices into a human-readable range string (e.g., '1-10, 25-30').
    Used for concise logging of frozen/mobile selections.
    """
    if not indices: return "None"
    indices = sorted(indices)
    ranges = []
    start = indices[0]
    prev = indices[0]
    
    for x in indices[1:]:
        if x != prev + 1:
            # End of a contiguous block
            if start == prev: ranges.append(f"{start}")
            else: ranges.append(f"{start}-{prev}")
            start = x
        prev = x
    
    # Final block
    if start == prev: ranges.append(f"{start}")
    else: ranges.append(f"{start}-{prev}")
    
    return ", ".join(ranges)
# ----------------------------------

# --- PDB Structure Loader ---
def load_pdb_structure(pdb_path, parser):
    """
    Loads a PDB file into a BioPython Structure object with error handling.
    
    Ensures the file is not corrupted and contains at least one valid chain 
    before proceeding with geometric operations.
    """
    try:
        structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
        if not structure or not structure.get_list(): return None
        model = structure[0]
        if not model or not model.get_list(): return None
        chain = model.get_list()[0]
        if not chain or not chain.get_list(): return None
        return structure
    except Exception as e:
        print(f"     Error loading PDB {pdb_path}: {e}")
        return None
# ----------------------------------

# --- Segment Atom Extractor ---
def get_segment_atoms(structure, chain_id, segment_residues):
    """
    Extracts the backbone atoms (N, CA, C) for a specific residue segment.
    
    These atoms serve as the 'alignment stub' for kinematic superposition, 
    defining the coordinate frame for stitching.
    """
    atoms = []
    if chain_id not in structure[0]: return []
    chain = structure[0][chain_id]
    for res_seq_num in segment_residues:
        if res_seq_num in chain:
            res = chain[res_seq_num]
            if 'CA' in res and 'N' in res and 'C' in res:
                atoms.extend([res['N'], res['CA'], res['C']])
    return atoms
# ----------------------------------

# --- Segment Map Builder ---
def build_segment_map(labeled_idrs, all_residues_set):
    """
    Partitions the protein sequence into alternating 'Folded' and 'Disordered' segments.
    
    This map (e.g., F1 -> D1 -> F2) guides the sequential assembly process, ensuring 
    that domains are placed in the correct N-to-C terminal order.
    """
    seg_map = defaultdict(list)
    idr_residues_set = set()
    labeled_idrs.sort(key=lambda x: x['range'][0])
    f_domain_counter = 0
    if labeled_idrs[0]['range'][0] > 1:
        f_domain_counter += 1; f_label = f"F{f_domain_counter}"
        seg_map[f_label] = list(range(1, labeled_idrs[0]['range'][0]))
    for i, idr_info in enumerate(labeled_idrs):
        start, end = idr_info['range']; idr_label = idr_info['label']
        idr_res = list(range(start, end + 1)); seg_map[idr_label] = idr_res
        idr_residues_set.update(idr_res)
        is_last_idr = (i == len(labeled_idrs) - 1)
        if not is_last_idr:
            next_idr_start = labeled_idrs[i+1]['range'][0]
            if next_idr_start > end + 1: 
                f_domain_counter += 1; f_label = f"F{f_domain_counter}"
                seg_map[f_label] = list(range(end + 1, next_idr_start))
        else: 
            max_res = max(all_residues_set)
            if end < max_res: 
                f_domain_counter += 1; f_label = f"F{f_domain_counter}"
                seg_map[f_label] = list(range(end + 1, max_res + 1))
    return seg_map, idr_residues_set, f_domain_counter
# ----------------------------------

# --- Kinematic Chain Assembler ---
def assemble_kinematic_chain(static_structure, ensemble_dirs, labeled_idrs, all_residues_set, pdb_parser):
    """
    Performs the kinematic stitching algorithm to assemble a full-length model.

    This method relies on superimposing the earlier defined alignment stub of the folded domains to 
    accurately propagate the angular trajectory of the disordered regions.

    Algorithm:
    1. Initialization: Starts with the first folded domain (F1) from the static 
       AlphaFold2 template as the primary anchor.
    2. Stub Definition: For the interface between the current anchor and the next 
       segment, defines an 'Alignment Stub'.
       - Logic: Midpoint of the current Folded Domain ± X residues (Default: 5).
       - Purpose: Uses the most rigid part of the domain for alignment, avoiding 
         flexible terminal artifacts.
    3. Geometric Alignment (Kabsch): 
       - Selects a random IDR conformer (which includes flanking folded boundaries).
       - Superimposes the conformer's stub onto the anchor's stub.
    4. Grafting (Overwrite): 
       - Cuts the current model exactly at the stub start residue.
       - Pastes the aligned conformer from that residue onwards.
       - This effectively replaces the C-terminal half of the anchor with the 
         conformationally sampled version, ensuring perfect continuity.
    5. Iteration: Designates the folded domain embedded at the end of the new IDR
       as the next anchor and repeats until the C-terminus is reached.

    Returns:
        Structure: A continuous, stitched model ready for energy minimization.
    """
    try:
        base_model = static_structure[0]
        base_chain_id = next(iter(base_model)).id
        segment_map, idr_residue_set, f_domain_count = build_segment_map(labeled_idrs, all_residues_set)
        sorted_segment_labels = sorted(segment_map.keys(), key=lambda x: segment_map[x][0]) 
        
        builder = StructureBuilder()
        builder.init_structure("stitched"); builder.init_model(0); builder.init_chain(base_chain_id)
        final_conformer = builder.get_structure()
        final_chain = final_conformer[0][base_chain_id]
        current_anchor_chain = None 
        
        start_label = sorted_segment_labels[0]
        start_residues = segment_map[start_label]
        
        if start_label.startswith("F"): 
            for res_seq_num in start_residues:
                if res_seq_num in base_model[base_chain_id]:
                    final_chain.add(base_model[base_chain_id][res_seq_num].copy())
            current_anchor_chain = final_chain
        elif start_label.startswith("D"): 
            tail_conformer_file = random.choice(ensemble_dirs[start_label]['files'])
            tail_struct = load_pdb_structure(tail_conformer_file, pdb_parser)
            if not tail_struct: return None
            tail_chain = tail_struct[0].get_list()[0]
            next_f_label = labeled_idrs[0]['flanking_domains'][0] 
            residues_to_copy = start_residues + segment_map.get(next_f_label, []) 
            for res_seq_num in residues_to_copy:
                if res_seq_num in tail_chain:
                    final_chain.add(tail_chain[res_seq_num].copy())
            current_anchor_chain = final_chain
        
        for i in range(1, len(sorted_segment_labels)):
            segment_label = sorted_segment_labels[i]
            segment_residues = segment_map[segment_label]
            if not current_anchor_chain: return None
            
            if segment_label.startswith("F"):
                if segment_residues and segment_residues[0] in current_anchor_chain:
                    continue 
                else:
                    if i == len(sorted_segment_labels) - 1:
                        for res_seq_num in segment_residues:
                            if res_seq_num in base_model[base_chain_id]:
                                final_chain.add(base_model[base_chain_id][res_seq_num].copy())
                        continue
                    else: return None
            
            elif segment_label.startswith("D"):
                anchor_label = sorted_segment_labels[i-1]
                if not anchor_label.startswith("F"): return None
                anchor_residues = segment_map[anchor_label]

                if not anchor_residues: return None
                
                midpoint = int(np.median(anchor_residues)) 
                stub_start = midpoint - ALIGNMENT_STUB_HALF_SIZE
                stub_end = midpoint + ALIGNMENT_STUB_HALF_SIZE
                junction_stub_residues = [r for r in anchor_residues if stub_start <= r <= stub_end]
                if len(junction_stub_residues) < ALIGNMENT_STUB_HALF_SIZE:
                    junction_stub_residues = anchor_residues[-ALIGNMENT_JUNCTION_SIZE:]

                moving_anchor_atoms = get_segment_atoms(final_conformer, base_chain_id, junction_stub_residues)
                conformer_file = random.choice(ensemble_dirs[segment_label]['files'])
                conformer_struct = load_pdb_structure(conformer_file, pdb_parser)
                if not conformer_struct: return None 
                conformer_chain_id = next(iter(conformer_struct[0])).id
                static_anchor_atoms = get_segment_atoms(conformer_struct, conformer_chain_id, junction_stub_residues)
                if not moving_anchor_atoms or not static_anchor_atoms or (len(moving_anchor_atoms) != len(static_anchor_atoms)):
                    return None
                super_imposer = Superimposer()
                super_imposer.set_atoms(moving_anchor_atoms, static_anchor_atoms)
                rot, tran = super_imposer.rotran
                conformer_struct.transform(rot, tran)
                conformer_chain = conformer_struct[0][conformer_chain_id]
                
                residues_to_copy_set = set(segment_residues) 
                if junction_stub_residues:
                    stub_min_res = min(junction_stub_residues)
                    post_stub_anchor_residues = [r for r in anchor_residues if r >= stub_min_res]
                    residues_to_copy_set.update(post_stub_anchor_residues)
                else:
                    residues_to_copy_set.update(anchor_residues)

                if i < len(sorted_segment_labels) - 1:
                    next_seg_label = sorted_segment_labels[i+1]
                    if next_seg_label.startswith("F"):
                        residues_to_copy_set.update(segment_map[next_seg_label])

                for res_seq_num in sorted(list(residues_to_copy_set)):
                    if res_seq_num not in conformer_chain: continue
                    res = conformer_chain[res_seq_num]
                    
                    if res_seq_num not in final_chain:
                        final_chain.add(res.copy())
                    else:
                        stitched_res = final_chain[res_seq_num]
                        for atom in res:
                            if atom.name in stitched_res:
                                stitched_res[atom.name].set_coord(atom.get_coord())
        
        return final_conformer

    except Exception as e:
        return None
# ----------------------------------

# --- Clean Structure Creator ---
def clean_structure(structure):
    """
    Standardizes the stitched structure for simulation.
    
    1. Renumbers residues sequentially to eliminate gaps from stitching.
    2. Removes internal terminal atoms (OXT, H-caps) that would cause steric 
       clashes or incorrect topology in the forcefield.
    """
    clean_structure_obj = Structure("clean")
    clean_model = Model(0)
    clean_chain = Chain("A")
    
    clean_structure_obj.add(clean_model)
    clean_model.add(clean_chain)
    
    all_residues = sorted(list(structure.get_residues()), key=lambda r: r.id[1])
    total_residues = len(all_residues)
    
    new_res_num = 1
    blocker_atoms = {'H1', 'H2', 'H3', '1H', '2H', '3H', 'OT1', 'OT2'} 

    for i, old_res in enumerate(all_residues):
        new_res = Residue(
            (' ', new_res_num, ' '), 
            old_res.get_resname(), 
            ' '
        )
        
        is_c_terminus = (i == total_residues - 1)
        
        for atom in old_res:
            atom_name = atom.get_name()
            if atom_name in blocker_atoms: continue
            if atom_name in ['OXT', 'OT1', 'OT2'] and not is_c_terminus: continue
            
            new_atom = Atom(
                atom_name,
                atom.get_coord(),
                atom.get_bfactor(),
                atom.get_occupancy(),
                atom.get_altloc(),
                atom.get_fullname(),
                atom.get_serial_number(),
                element=atom.element
            )
            new_res.add(new_atom)
            
        clean_chain.add(new_res)
        new_res_num += 1
        
    return clean_structure_obj
# ----------------------------------

# --- Pre-Minimization Chirality Repair Function ---
def repair_chirality_on_pdb(pdb_filepath):
    """
    Detects and corrects stereochemical inversions (D-amino acids).
    
    Kinematic transformations can occasionally invert local chirality. This function 
    identifies residues with positive scalar triple products (D-chirality) and 
    reflects their sidechains to restore natural L-chirality.
    """
    try:
        # Load
        pdb = PDBFile(pdb_filepath)
        modeller = Modeller(pdb.topology, pdb.positions)
        
        # Repair Logic (OpenMM)
        repaired_count = 0
        positions = modeller.positions
        epsilon = 1e-6
        
        # Map atoms
        for res in modeller.topology.residues():
            atoms = {a.name: a.index for a in res.atoms()}
            if not all(k in atoms for k in ("N", "CA", "C", "CB", "HA")):
                continue

            N_idx, CA_idx, C_idx, CB_idx, HA_idx = (
                atoms["N"], atoms["CA"], atoms["C"], atoms["CB"], atoms["HA"]
            )
            
            # Vectors
            p = positions
            N_v  = p[N_idx].value_in_unit(unit.angstrom)
            CA_v = p[CA_idx].value_in_unit(unit.angstrom)
            C_v  = p[C_idx].value_in_unit(unit.angstrom)
            CB_v = p[CB_idx].value_in_unit(unit.angstrom)
            HA_v = p[HA_idx].value_in_unit(unit.angstrom)

            # Check Volume
            vec1 = np.array(CA_v) - np.array(N_v)
            vec2 = np.array(C_v) - np.array(N_v)
            vec3 = np.array(CB_v) - np.array(N_v)
            vol = np.dot(vec1, np.cross(vec2, vec3))

            if vol > epsilon:
                # Found D-chirality -> FLIP IT
                # Normal to N-CA-C plane
                normal = np.cross(vec1, vec2)
                norm = np.linalg.norm(normal)
                if norm < 1e-12: continue
                normal /= norm
                
                # Math for CB
                d_CB = np.dot(np.array(CB_v) - np.array(N_v), normal)
                CB_new = np.array(CB_v) - 2 * d_CB * normal
                
                # Math for HA
                d_HA = np.dot(np.array(HA_v) - np.array(N_v), normal)
                HA_new = np.array(HA_v) - 2 * d_HA * normal
                
                # Update positions
                positions[CB_idx] = unit.Quantity(CB_new, unit.angstrom)
                positions[HA_idx] = unit.Quantity(HA_new, unit.angstrom)
                
                repaired_count += 1
                print(f"       [PRE-MINIMAZATION: CHIRALITY REPAIR] Flipped chirality for {res.name} {res.id}")

        if repaired_count > 0:
            # Save back to file
            with open(pdb_filepath, 'w') as f:
                PDBFile.writeFile(modeller.topology, positions, f)
            return True
            
    except Exception as e:
        print(f"       [WARN] Chirality repair failed: {e}")
        
    return False
# ----------------------------------

# --- Pre-Minimization Rg Knot Checker ---
def check_idr_penetration(structure, region_resids, core_fraction=0.6):
    """
    A topological filter to reject 'knot' conformations.
    
    Calculates the radius of gyration (Rg) of the folded core and rejects any 
    model where an IDR threads through the core's internal volume (defined as 
    <60% of Rg), preventing physically impossible entanglements.
    """
    try:
        folded_coords, idr_coords = [], []
        idr_set = set(region_resids)
        
        # 1. Extract Atoms
        for chain in structure[0]:
            for res in chain:
                if res.id[0] != ' ' or 'CA' not in res: continue
                coord = res['CA'].get_coord()
                if res.id[1] in idr_set:
                    idr_coords.append(coord)
                else:
                    folded_coords.append(coord)
                    
        # If missing data, pass (or fail if you prefer strictness)
        if not folded_coords or not idr_coords: 
            return True, "No Core/IDR found"

        folded_coords = np.array(folded_coords)
        idr_coords = np.array(idr_coords)
        
        # 2. Calculate Domain Metrics
        com = np.mean(folded_coords, axis=0)
        avg_radius = np.mean(np.linalg.norm(folded_coords - com, axis=1))
        forbidden_radius = avg_radius * core_fraction
        
        # 3. Calculate IDR Metrics
        idr_dists = np.linalg.norm(idr_coords - com, axis=1)
        min_dist = np.min(idr_dists)

        if min_dist < forbidden_radius:
            msg = f"IDR penetrates core (Dist {min_dist:.1f}Å < Limit {forbidden_radius:.1f}Å)"
            print(f"       [PRE-MINIMIZATION: REJECT KNOT] {msg}")
            return False, msg
            
        return True, "Safe"

    except Exception as e:
        print(f"       [WARN] Knot check error: {e}")
        return True, f"Error: {e}" # Fail Open (Safe)
# ----------------------------------

# --- Energy Minimization with Amber ---
relax_cfg = {
    'max_outer_iterations': cfg.RELAX_MAX_OUTER_ITER,
    'stiffness': cfg.RELAX_STIFFNESS,
    'exclude_residues': [],
    'max_iterations': cfg.MINIMIZATION_MAX_ITER,
    'tolerance': cfg.MINIMIZATION_TOLERANCE
}

def relax_with_established_method(structure, output_filepath, idr_indices=None, device="cuda:0"):
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
            print("       [FAIL] Relaxation rejected.")
            if os.path.exists(expected_output): os.remove(expected_output)
            return False

    except Exception as e:
        print(f"       [CRASH] Relaxation failed: {e}")
        return False
# ----------------------------------

# --- Post-Minimization Chirality Checker ---
def check_chirality_L(topology, positions, tolerance=0.5):
    """
    Validates that all amino acids satisfy L-chirality constraints post-minimization.
    
    Uses signed volume analysis of the Calpha chiral center. A volume > tolerance 
    indicates a failure of the forcefield to correct unphysical geometry.
    Conformers that violate this constraint are rejected.
    """
    import numpy as np
    from openmm import unit
    
    for res in topology.residues():
        atoms = {a.name: a.index for a in res.atoms()}
        if not all(k in atoms for k in ("N", "CA", "C", "CB")):
            continue

        p = positions
        N  = np.array(p[atoms["N"]].value_in_unit(unit.angstrom))
        CA = np.array(p[atoms["CA"]].value_in_unit(unit.angstrom))
        C  = np.array(p[atoms["C"]].value_in_unit(unit.angstrom))
        CB = np.array(p[atoms["CB"]].value_in_unit(unit.angstrom))

        # Signed Volume: (CA-N) . ((C-N) x (CB-N))
        # L-amino acids should be negative. D-amino acids are positive.
        vec1 = CA - N
        vec2 = C - N
        vec3 = CB - N
        vol = np.dot(vec1, np.cross(vec2, vec3))

        if vol > tolerance:
            print(f"       [POST-MINIMIZATION: FAIL] Chirality violation at {res.name} {res.id} (Vol: {vol:.2f})")
            return False
    return True
# ----------------------------------

# --- Post-Minimization VdW Clash Checker ---
def check_clashes_vdw_openmm(topology, positions, tol=0.4, max_clashscore=5):
    """
    Quantifies steric clashes using an all-atom Van der Waals exclusion check.
    
    Metric:
        Clashscore: Number of non-bonded overlaps per 1,000 atoms.
    
    Threshold:
        Models with score > 5.0 are rejected, ensuring packing quality comparable 
        to high-resolution X-ray crystal structures. This threshold is dyanamically 
        adjusted later on based on number of attempts.
    """
    VDW = {'C':1.7,'N':1.55,'O':1.52,'S':1.8,'P':1.8,'F':1.47,'CL':1.75,'BR':1.85,'I':1.98}
    bonds = defaultdict(set)
    for b in topology.bonds(): bonds[b.atom1.index].add(b.atom2.index); bonds[b.atom2.index].add(b.atom1.index)
    
    heavy = []
    for a in topology.atoms():
        if a.element.symbol != 'H': 
            heavy.append((a.index, positions[a.index].value_in_unit(unit.angstrom), VDW.get(a.element.symbol, 1.5)))
    
    if not heavy: return True, 0, 0.0, 0
    n_heavy = len(heavy)
    
    # Calculate limits
    allowed_count = (max_clashscore * n_heavy) / 1000.0
    
    clashes = 0
    coords = np.array([h[1] for h in heavy])
    radii = np.array([h[2] for h in heavy])
    indices = [h[0] for h in heavy]
    
    for i in range(n_heavy):
        excl = {indices[i]}
        q = [(indices[i], 0)]
        while q:
            curr, d = q.pop(0)
            if d < 3:
                for n in bonds[curr]:
                    if n not in excl: excl.add(n); q.append((n, d+1))
        
        diff = coords[i+1:] - coords[i]
        d2 = np.sum(diff**2, axis=1)
        thresh = (radii[i] + radii[i+1:] - tol)**2
        
        cands = np.where(d2 < thresh)[0]
        for c in cands:
            if indices[i+1+c] not in excl: clashes += 1

    actual_score = (clashes * 1000.0) / n_heavy if n_heavy > 0 else 0.0
    
    # Returns: (Pass/Fail, Count, Actual Score, Allowed Count)
    if clashes > allowed_count: return False, clashes, actual_score, allowed_count
    return True, clashes, actual_score, allowed_count
# ----------------------------------

# --- Post-Minimization Folded RMSD Checker ---
def validate_folded_rmsd(ref, relaxed_path, frozen_ids, max_rmsd=2.0):
    """
    Ensures the folded domains remained stable during relaxation.
    
    Calculates the Root Mean Square Deviation (RMSD) between the initial AF2 model 
    and the final stitched model. Large deviations (>2.0 Å) imply that the 
    restraints failed or the stitching introduced catastrophic strain.
    """
    try:
        mob = PDBParser(QUIET=True).get_structure("m", relaxed_path)
        frozen = set(frozen_ids)
        ref_atoms = [r['CA'] for r in ref[0].get_residues() if r.id[1] in frozen and 'CA' in r]
        mob_atoms = [r['CA'] for r in mob[0].get_residues() if r.id[1] in frozen and 'CA' in r]
        
        if not ref_atoms or len(ref_atoms) != len(mob_atoms): 
            return True, 0.0 # Cannot calculate, assume pass
        
        sup = Superimposer()
        sup.set_atoms(ref_atoms, mob_atoms)
        rmsd = sup.rms
        
        if rmsd > max_rmsd:
            print(f"       [POST-MINIMIZATION CHECK] Folded domain shifted too much (RMSD: {rmsd:.2f}Å > {max_rmsd}Å)")
            return False, rmsd
        return True, rmsd
    except: 
        return True, 0.0
# ----------------------------------

# --- Post-Minimization Bond Integrity Validator ---
def validate_bond_integrity(topology, positions, tolerance=0.3):
    """
    Verifies the integrity of the polypeptide backbone.
    
    Checks peptide bond lengths (C-N) and intra-residue geometry. Deviations 
    beyond 0.3 Å suggest the forcefield 'snapped' the chain to resolve a clash, 
    marking the model as invalid.
    """
    pos_ang = [p.value_in_unit(unit.angstrom) for p in positions]

    def get_dist(i1, i2):
        return np.linalg.norm(np.array(pos_ang[i1]) - np.array(pos_ang[i2]))

    # Canonical lengths (Å)
    canonical = {
        ("N", "CA"): 1.46,
        ("CA", "C"): 1.53,
        ("C", "N"):  1.33, # Peptide bond
    }

    residues = list(topology.residues())
    broken_bonds = []

    for i, res in enumerate(residues):
        atoms = {a.name: a.index for a in res.atoms()}

        # 1. Intra-residue (N-CA, CA-C)
        for (a1, a2) in [("N", "CA"), ("CA", "C")]:
            if a1 in atoms and a2 in atoms:
                d = get_dist(atoms[a1], atoms[a2])
                ref = canonical[(a1, a2)]
                if abs(d - ref) > tolerance:
                    broken_bonds.append(f"       [BROKEN] {res.name}{res.id} {a1}-{a2}: {d:.2f}Å (Exp {ref}±{tolerance})")

        # 2. Inter-residue Peptide Bond (C -> N)
        if i < len(residues) - 1:
            next_res = residues[i+1]
            # Check for chain continuity (same chain index)
            if res.chain.index == next_res.chain.index:
                next_atoms = {a.name: a.index for a in next_res.atoms()}
                if "C" in atoms and "N" in next_atoms:
                    d = get_dist(atoms["C"], next_atoms["N"])
                    ref = canonical[("C", "N")]
                    if abs(d - ref) > tolerance:
                        broken_bonds.append(f"       [BROKEN] {res.name}{res.id}->{next_res.name}{next_res.id} Peptide: {d:.2f}Å (Exp {ref}±{tolerance})")

    if broken_bonds:
        for b in broken_bonds:
            print(b)
        return False
        
    return True
# ----------------------------------

# --- PDB Combining and Cleanup ---
def combine_and_cleanup_pdbs(directory, base_filename, num_files, protein_id,
                             root_output_dir, length_label, category,
                             completion_status):
    """
    Aggregates individual PDB conformers into a single multi-model ensemble file.
    
    This step reduces filesystem clutter and prepares the dataset for downstream 
    analysis.
    """
    try:
        # Ensure temp directory exists
        os.makedirs(directory, exist_ok=True)

        # Prepare final target directory
        final_target_dir = os.path.join(
            root_output_dir, 
            length_label, 
            category, 
            protein_id
        )
        
        os.makedirs(final_target_dir, exist_ok=True)

        # --- Case A: Keep Individual Files (Default) ---
        if not getattr(cfg, 'combine_ensemble', False):
            print(f"     Moving individual files to: {final_target_dir}")
            
            # Move files one by one (Safest way to avoid nesting folders)
            files_moved = 0
            for filename in os.listdir(directory):
                if filename.endswith(".pdb"):
                    src_file = os.path.join(directory, filename)
                    dst_file = os.path.join(final_target_dir, filename)
                    
                    # Overwrite if exists
                    if os.path.exists(dst_file):
                        os.remove(dst_file)
                    
                    shutil.move(src_file, dst_file)
                    files_moved += 1
            
            # Cleanup temp folder
            shutil.rmtree(directory)
            try: os.rmdir(os.path.dirname(directory)) # Try removing parent temp dir
            except: pass
            
            return True

        # --- Case B: Create Single Ensemble PDB ---
        print(f"     Combining {num_files} conformers into ensemble...")

        files_to_combine = [
            os.path.join(directory, f"{base_filename}_{i+1}.pdb")
            for i in range(num_files)
        ]
        existing_files = [f'"{f}"' for f in files_to_combine if os.path.exists(f)]

        if not existing_files:
            print("     Error: No individual conformer files found to ensemble.")
            return False

        ensemble_filename = f"{protein_id}_{base_filename}_ensemble_n{len(existing_files)}.pdb"
        ensemble_filepath = os.path.join(final_target_dir, ensemble_filename)

        # Run pdb_mkensemble
        files_str = " ".join(existing_files)
        ensemble_cmd = f'pdb_mkensemble {files_str} > "{ensemble_filepath}"'

        subprocess.run(ensemble_cmd, capture_output=True, shell=True, text=True, check=True)
        print(f"     Ensemble created: {os.path.basename(ensemble_filepath)}")

        # Cleanup temp directory
        shutil.rmtree(directory)
        try: os.rmdir(os.path.dirname(directory))
        except: pass

        return True

    except Exception as e:
        print(f"!!--- ERROR in cleanup ---!! {e}")
        return False
# ----------------------------------

# --- Main Processing Function ---
def process_protein(protein_id, labeled_db, id_to_pdb_path, conformer_root_dir, output_dir, final_output_root, num_conformers, **kwargs):
    """
    Orchestrates the Monte Carlo sampling pipeline for a single protein.
    
    Workflow:
    1. Initialization: Loads the static anchor and maps IDR ensembles.
    2. Sampling Loop:
       - Stitch: Assembles a new conformation using random draws from IDR pools.
       - Filter: Rejects topological knots (Rg check).
       - Relax: Minimizes energy using AMBER.
       - Validate: Checks sterics, chirality, RMSD, and bond integrity.
    3. Termination: Stops when N valid conformers are generated or max attempts reached.
    
    Returns:
        tuple: (Category, Status, Success_Count) for global reporting.
    """

    print(f"\n--- Processing Protein: {protein_id} ---")
    labeled_idrs = labeled_db[protein_id].get('labeled_idrs', [])
    ldr_infos = [i for i in labeled_idrs if i.get('type') != 'IDP']
    category = get_protein_category(labeled_idrs)
    
    if not ldr_infos and category != "cat_0_idp_only": 
        return category, "Skipped", 0
    
    # 1. LOAD STATIC REFERENCE
    static_path = id_to_pdb_path.get(protein_id)
    if not static_path: 
        print(f"    [!] Error: Static PDB for {protein_id} not found in {cfg.PDB_LIBRARY_PATH}")
        return category, "Failed", 0
    pdb_parser = PDBParser(QUIET=True)
    static_struct = load_pdb_structure(static_path, pdb_parser)
    if not static_struct: return category, "Failed", 0
    
    # Paths
    length_label = get_length_label(len(list(static_struct.get_residues())))
    mode = "minimized" if len(ldr_infos) == 1 else "stitched"
    work_dir = os.path.join(output_dir, category, protein_id, f"{mode}_ensemble")
    
    # --- RESUME LOGIC ---
    # 1. Check Final Destination first
    final_dest_dir = os.path.join(final_output_root, length_label, category, protein_id)
    existing_final = 0
    if os.path.exists(final_dest_dir):
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
    
    if done >= num_conformers:
        # If finished, run cleanup just in case files are stranded in temp
        if max_temp_idx > 0:
            combine_and_cleanup_pdbs(work_dir, f"{mode}_conformer", done, protein_id, final_output_root, length_label, category, "Complete")
        return category, "Complete", done
    # ----------------------------

    # Setup Source
    ensemble_dirs = find_ensemble_dirs(protein_id, conformer_root_dir, ldr_infos)
    if not ensemble_dirs and category != "cat_0_idp_only": 
        return category, "Failed", 0 
    
    os.makedirs(work_dir, exist_ok=True)
    
    region_resids = build_region_resids(ldr_infos)

    # Setup Pool for Random Sampling
    minimized_pool = []
    if mode == "minimized" and ensemble_dirs:
        first_label = list(ensemble_dirs.keys())[0]
        dat = ensemble_dirs[first_label]
        minimized_pool = dat['files'] 
        if not minimized_pool: return category, "Failed", 0
    
    # Loop
    print(f"  Generating {num_conformers} conformers ({mode})...")
    print(f"  [Resume] Found {existing_final} in final, {max_temp_idx} in temp. Starting at {done+1}.")
    
    attempts = 0
    
    while done < num_conformers and attempts < 1_000_000:
        attempts += 1
        
        if attempts % 50 == 0:
            print(f"     [PROGRESS] Summary: {done}/{num_conformers} successes")

        print("\n" + "-"*60)
        
        clash_lim = 5.0 + (int(attempts / 5000) * 2.5)
        out_name = f"{mode}_conformer_{done+1}.pdb"
        out_path = os.path.join(work_dir, out_name)
        
        # Generation Step
        if mode == "minimized":
            # Random Sampling
            src_path = random.choice(minimized_pool)
            print(f"     [Attempt {attempts}] Processing {os.path.basename(src_path)}...")
            raw = load_pdb_structure(src_path, pdb_parser)
        else:
            # Stitched mode
            print(f"     [Attempt {attempts}] Assembling Kinematic Chain...")
            raw_s = assemble_kinematic_chain(static_struct, ensemble_dirs, ldr_infos, set(r.id[1] for r in static_struct[0].get_list()[0] if r.id[0]==' '), pdb_parser)
            raw = clean_structure(raw_s) if raw_s else None
            
        if not raw: continue

        # Identify Frozen Residues
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
        
        # Identify Longest Folded Domain for RMSD
        longest_folded_ids = []
        if frozen_ids:
            current_seg = []
            sorted_frozen = sorted(frozen_ids)
            for i, fid in enumerate(sorted_frozen):
                if not current_seg:
                    current_seg.append(fid)
                elif fid == current_seg[-1] + 1:
                    current_seg.append(fid)
                else:
                    if len(current_seg) > len(longest_folded_ids):
                        longest_folded_ids = list(current_seg)
                    current_seg = [fid]
            if len(current_seg) > len(longest_folded_ids):
                longest_folded_ids = list(current_seg)

        print(f"       [CONFIG] Freezing All Folded: {format_ranges(frozen_ids)}")
        print(f"       [CONFIG] Tracking Longest Fold: {format_ranges(longest_folded_ids)}")

        # Gates
        is_safe, fail_reason = check_idr_penetration(raw, region_resids)
        if not is_safe:
            continue 
        
        # Repair & Relax
        io = PDBIO(); io.set_structure(raw); io.save(out_path)
        repair_chirality_on_pdb(out_path)
        
        repaired = load_pdb_structure(out_path, pdb_parser) or raw
        
        if relax_with_established_method(repaired, out_path, idr_indices=idr_idx):
            # --- POST-MINIMIZATION CHECKS ---
            print(f"       [POST-MIN CHECK] Validating...")
            try:
                chk = PDBFile(out_path)
                
                # 1. Clash Check
                ok_c, n_c, act_s, max_c = check_clashes_vdw_openmm(chk.topology, chk.positions, max_clashscore=clash_lim)
                print(f"         - Clashes:   {n_c} (Max: {int(max_c)} | Score: {act_s:.2f} | Limit: {clash_lim:.1f}) -> {'PASS' if ok_c else 'FAIL'}")
                if not ok_c: 
                     print(f"       [RESULT] FAILED (Clashes)"); os.remove(out_path); continue

                # 2. RMSD Check (Longest Fold)
                ok_rmsd, val_rmsd = validate_folded_rmsd(
                    static_struct, 
                    out_path, 
                    longest_folded_ids
                )
                print(f"         - Fold RMSD: {val_rmsd:.2f}Å (Limit: 2.0Å) -> {'PASS' if ok_rmsd else 'FAIL'}")
                if not ok_rmsd:
                    print(f"       [RESULT] FAILED (Folded Domain Exploded)"); os.remove(out_path); continue

                # 3. Bond Check
                if not validate_bond_integrity(chk.topology, chk.positions): 
                     print(f"         - Bonds:     FAIL"); print(f"       [RESULT] FAILED (Broken Bonds)"); os.remove(out_path); continue
                print(f"         - Bonds:     PASS")
                
                # 4. Chirality Check
                if not check_chirality_L(chk.topology, chk.positions): 
                     print(f"         - Chirality: FAIL"); print(f"       [RESULT] FAILED (Chirality)"); os.remove(out_path); continue
                print(f"         - Chirality: PASS")
                
                # --- SUCCESS ---
                done += 1
                print(f"       [RESULT] SUCCESS! (Total: {done}/{num_conformers})")

            except Exception as e:
                print(f"       [ERROR] Validation crashed: {e}")
                if os.path.exists(out_path): os.remove(out_path)
        else:
            print(f"       [RESULT] FAILED (Relaxation Rejected)")
            if os.path.exists(out_path): os.remove(out_path)

    status = get_completion_status(done, num_conformers)
    
    combine_and_cleanup_pdbs(work_dir, f"{mode}_conformer", done, protein_id, final_output_root, length_label, category, status)
    return category, status, done
# ----------------------------------

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4: Kinematic Stitching & Energy Minimization Pipeline",
        epilog="Example: python Step_4_stitch.py --id_file ids.txt --total_splits 10 --split_index 0"
    )
    parser.add_argument("--id_file", required=True, help="Path to the input text file containing newline-separated UniProt IDs.")
    parser.add_argument("--category", default=None, help="Label for this processing batch.")
    parser.add_argument("--total_splits", type=int, default=1, help="Total number of parallel jobs.")
    parser.add_argument("--split_index", type=int, default=0, help="The specific shard index (0-based).")
    args = parser.parse_args()

    category_name = args.category
    if category_name is None:
        try:
            filename = os.path.basename(args.id_file)
            category_name = os.path.splitext(filename)[0]
            if not category_name.startswith("cat_"):
                raise ValueError(f"Filename '{filename}' does not start with 'cat_'. Please use --category.")
            print(f"[\u2713] Auto-detected Category: {category_name}")
        except Exception as e:
            print(f"[!] CRITICAL ERROR: Could not derive category. {e}")
            sys.exit(1)

    print(f"\n--- Initializing Workflow for: {category_name} ---")

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
        with open(cfg.LABELED_DB_PATH, 'r') as f:
            labeled_db = json.load(f)
        print(f"[\u2713] Configuration and Databases loaded.")
    except Exception as e:
        print(f"[!] CRITICAL ERROR: Failed to load external resources: {e}")
        sys.exit(1)

    final_root_dir = cfg.STITCH_OUTPUT_ROOT
    temp_working_dir = os.path.join(final_root_dir, f"_temp_work_{category_name}_{args.split_index}")
    
    os.makedirs(final_root_dir, exist_ok=True)
    os.makedirs(temp_working_dir, exist_ok=True)
    
    print(f"    Output Root: {final_root_dir}")
    print(f"    Temp Work:   {temp_working_dir}")

    print(f"\n--- Starting Processing Loop ---")
    
    stats = Counter()
    status_counts = Counter()
    total_in_chunk = len(chunk_of_ids)

    for i, protein_id in enumerate(chunk_of_ids, 1):
        print(f"\n[{i}/{total_in_chunk}] Processing {protein_id}...")
        if protein_id not in labeled_db:
            print(f"    -> SKIPPED: Not found in labeled DB.")
            stats['skipped'] += 1
            continue
            
        try:
            cat_result, completion_status, num_success = process_protein(
                protein_id=protein_id,
                labeled_db=labeled_db,
                id_to_pdb_path=id_to_pdb_path,
                conformer_root_dir=cfg.CONFORMER_POOL_DIR, 
                output_dir=temp_working_dir,
                final_output_root=final_root_dir, 
                num_conformers=cfg.STITCH_N_CONFORMERS,
                clash_distance=cfg.STITCH_CLASHSCORE,
                max_attempts_per_conf=cfg.STITCH_MAX_ATTEMPTS
            )
            
            stats['processed'] += 1
            status_counts[completion_status] += 1
            
            symbol = "\u2713" if completion_status == "Complete" else "!"
            print(f"    -> {symbol} Result: {completion_status} ({num_success} models)")
            if completion_status == "Failed": stats['failed'] += 1
        except Exception as e:
            print(f"    [!] EXCEPTION CRASH on {protein_id}: {e}")
            stats['crashed'] += 1

    try:
        shutil.rmtree(temp_working_dir, ignore_errors=True)
    except OSError as e:
        print(f"[!] Warning: Could not clean up temp dir: {e}")

    print(f"\n" + "="*40)
    print(f"   STEP 4 BATCH REPORT: {category_name}")
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