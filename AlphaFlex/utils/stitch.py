"""Helper functions for Step 4 kinematic stitching and relaxation."""
import os
import glob
import re
import random
import numpy as np
from collections import defaultdict

from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.Structure import Structure

import config as cfg

# --- Constants ---
ALIGNMENT_STUB_HALF_SIZE = cfg.ALIGNMENT_STUB_HALF_SIZE
ALIGNMENT_JUNCTION_SIZE = cfg.ALIGNMENT_JUNCTION_SIZE
MIN_CONFORMER_POOL_SIZE = cfg.MIN_CONFORMER_POOL_SIZE
# ---


# --- Completion Status Helper ---
def get_completion_status(conformer_count, target_conformers):
    """Return the success tier of the sampling process."""
    if conformer_count >= target_conformers:
        return "Complete"
    elif conformer_count > 50:
        return "Partially Complete"
    else:
        return "Failed"
# ----------------------------------

# --- Length Binning Helper ---
def get_length_label(num_res):
    """Stratify proteins into length bins by residue count."""
    if num_res <= 250: return "0-250"
    elif num_res <= 500: return "251-500"
    elif num_res <= 1000: return "501-1000"
    elif num_res <= 1500: return "1001-1500"
    elif num_res <= 2000: return "1501-2000"
    else: return "2001+"
# ----------------------------------

# --- IDR Region Identification Helper ---
def build_region_resids(labeled_idrs):
    """Aggregate all residue indices belonging to IDRs."""
    region_resids = []
    for idr in labeled_idrs:
        start, end = idr['range']
        region_resids.extend(range(start, end+1))
    return region_resids
# ----------------------------------

# --- PDB File Mapping Helper ---
def get_id_to_pdb_path():
    """Map UniProt IDs to local PDB file paths."""
    dbg_map = getattr(cfg, "DEBUG_PDB_MAP", False)
    if cfg.VERBOSE:
        print(f"--- Mapping PDB library: {cfg.PDB_LIBRARY_PATH}")

    id_to_pdb_path = {}

    if not os.path.exists(cfg.PDB_LIBRARY_PATH):
         print(f"[!] CRITICAL: Directory not found!")
         return {}

    files = os.listdir(cfg.PDB_LIBRARY_PATH)
    if cfg.VERBOSE:
        print(f"    Found {len(files)} files in library.")

    # Pre-compile regex patterns
    af_pattern = re.compile(r"AF-([A-Z0-9]+)-F1")
    simple_pattern = re.compile(r"([A-Z0-9]+)\.pdb")

    for filename in files:
        if not filename.endswith(".pdb"):
            if dbg_map:
                print(f"  [Skip] Not a .pdb: {filename}")
            continue

        pdb_path = os.path.join(cfg.PDB_LIBRARY_PATH, filename)

        # 1. Try AlphaFold Long Format
        match = af_pattern.search(filename)
        if match:
            pid = match.group(1)
            id_to_pdb_path[pid] = pdb_path
            if dbg_map:
                print(f"  [Match] AF-Format: {filename} -> ID: {pid}")
            continue

        # 2. Try Simple Format
        match_simple = simple_pattern.match(filename)
        if match_simple:
            pid = match_simple.group(1)
            id_to_pdb_path[pid] = pdb_path
            if dbg_map:
                print(f"  [Match] Simple-Format: {filename} -> ID: {pid}")
        else:
            if dbg_map:
                print(f"  [FAIL] No Regex Matched for: {filename}")

    print(f"--- Finished Mapping. Total IDs: {len(id_to_pdb_path)} ---")
    return id_to_pdb_path
# ----------------------------------

# --- Protein Category Helper ---
def get_protein_category(labeled_idrs):
    """Categorize a protein by IDR complexity (Loop > Linker > Tail)."""
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
def find_ensemble_dirs(protein_id, conformer_root_dir, labeled_idrs, verbose=False):
    """Locate the pre-generated conformer ensembles for each disordered segment."""
    ensemble_paths = {}
    base_protein_path = os.path.join(conformer_root_dir, protein_id)
    if not os.path.isdir(base_protein_path):
        if verbose:
            print(f"   Error: Base conformer directory for {protein_id} not found. Skipping.")
        return None
    found_all = True
    for idr_info in labeled_idrs:
        idr_type = idr_info.get("type"); idr_label_d = idr_info.get("label")
        if idr_type == "IDP": continue
        start, end = idr_info.get("range"); idr_label_range = f"idr_{start}-{end}"
        type_dir_name = idr_type.replace(" ", "_")
        idr_conformer_path = os.path.join(base_protein_path, type_dir_name, idr_label_range)
        pdb_files = glob.glob(os.path.join(idr_conformer_path, "*_validated.pdb"))

        if not pdb_files:
            old_path = os.path.join(base_protein_path, idr_label_range)
            pdb_files = glob.glob(os.path.join(old_path, "*_validated.pdb"))
            if pdb_files: idr_conformer_path = old_path
            else:
                if verbose:
                    print(f"     Error: No '*_validated.pdb' files found for {idr_label_d} ({idr_label_range}).")
                found_all = False; break

        if verbose and len(pdb_files) < MIN_CONFORMER_POOL_SIZE:
            print(f"     Warning: Ensemble for {idr_label_d} has only {len(pdb_files)} files.")

        ensemble_paths[idr_label_d] = {
            'path': idr_conformer_path, 'files': pdb_files, 'range': (start, end),
            'type': idr_type, 'flanking_domains': idr_info.get('flanking_domains', [])
        }
    return ensemble_paths if found_all else None
# ----------------------------------

# --- Range Identifier ---
def format_ranges(indices):
    """Format a list of residue indices into a range string (e.g., '1-10, 25-30')."""
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
def load_pdb_structure(pdb_path, parser, verbose=False):
    """Load a PDB file into a BioPython Structure object."""
    try:
        structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
        if not structure or not structure.get_list(): return None
        model = structure[0]
        if not model or not model.get_list(): return None
        chain = model.get_list()[0]
        if not chain or not chain.get_list(): return None
        return structure
    except Exception as e:
        if verbose:
            print(f"     Error loading PDB {pdb_path}: {e}")
        return None
# ----------------------------------

# --- Segment Atom Extractor ---
def get_segment_atoms(structure, chain_id, segment_residues):
    """Extract the backbone atoms (N, CA, C) for a residue segment."""
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
    """Partition the sequence into alternating folded and disordered segments."""
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

# --- Round-robin conformer draw ---
def _next_conformer(entry):
    """Draw the next conformer file from a sub-ensemble pool in round-robin order."""
    files = entry.get('files') or []
    if not files:
        return None
    order = entry.get('_order')
    cursor = entry.get('_cursor', 0)
    if not order or cursor >= len(order):
        order = list(files)
        random.shuffle(order)
        entry['_order'] = order
        cursor = 0
    entry['_cursor'] = cursor + 1
    return order[cursor]
# ----------------------------------

# --- Truncated sub-conformer back-numbering ---
def renumber_pool(src_dir, dst_dir, offset, verbose=False):
    """Write copies of every *_validated.pdb in src_dir with resSeq += offset."""
    os.makedirs(dst_dir, exist_ok=True)
    src = sorted(glob.glob(os.path.join(src_dir, "*_validated.pdb")))
    out = []
    for p in src:
        dst = os.path.join(dst_dir, os.path.basename(p))
        with open(p) as fh, open(dst, "w") as w:
            for line in fh:
                if line[:6].strip() in ("ATOM", "HETATM"):
                    try:
                        new = int(line[22:26]) + offset
                    except ValueError:
                        w.write(line); continue
                    line = line[:22] + f"{new:>4d}" + line[26:]
                w.write(line)
        out.append(dst)
    if verbose:
        print(f"   renumbered {len(out)} conformers by +{offset} -> {dst_dir}", flush=True)
    return out
# ----------------------------------

# --- Kinematic Chain Assembler ---
def assemble_kinematic_chain(static_structure, ensemble_dirs, labeled_idrs, all_residues_set, pdb_parser):
    """Assemble a full-length model by kinematic stitching of IDR conformers."""
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
            tail_conformer_file = _next_conformer(ensemble_dirs[start_label])
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
                conformer_file = _next_conformer(ensemble_dirs[segment_label])
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
    """Standardize the stitched structure for simulation."""
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
