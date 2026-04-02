"""
idpforge/utils/structure_repair.py

Pre-relaxation structural repairs for the IDPForge pipeline.

Core functionality:
  - Detecting and repairing D-amino acid chirality flips.
  - Fixing scrambled histidine imidazole ring atom naming after AMBER relaxation.
"""

import sys
import numpy as np
import numpy.linalg as LA

# ============================================================
# CONSTANTS
# ============================================================

# Atoms that are NOT reflected during chirality repair.
# Everything else in the residue (CB, CG, CD, ..., hydrogens) is sidechain.
_CHIRALITY_BACKBONE_ATOMS = {'N', 'H', 'CA', 'HA2', 'HA3', 'C', 'O', 'OXT'}

# ============================================================
# HISTIDINE RING NAMING CONSTANTS
# ============================================================
_HIS_VARIANTS = {'HIS', 'HID', 'HIE', 'HIP'}
_BACKBONE_NAMES = {'N', 'H', 'CA', 'HA', 'C', 'O', 'CB', 'HB2', 'HB3', 'OXT'}
_HIS_RING_NAMES = ['CG', 'ND1', 'CE1', 'NE2', 'CD2']
_RING_BOND_MIN = 1.25
_RING_BOND_MAX = 1.45
_H_BOND_MAX = 1.15
_HIS_H_MAP = {'ND1': 'HD1', 'CE1': 'HE1', 'NE2': 'HE2', 'CD2': 'HD2'}


def repair_chirality(pdb_path, verbose=False):
    """
    Detects D-amino acids using OpenMM topology (consistent with check_chirality
    in structure_validation.py) and reflects all sidechain atoms across the N-CA-C
    plane to restore L-isomer geometry.

    Args:
        pdb_path (str): Path to the PDB file to process (modified in-place).
        verbose (bool): If True, prints residues that required flipping.

    Returns:
        int: The total number of chiral centers repaired.
    """
    try:
        from openmm.app import PDBFile as OMMPDBFile
        from openmm import unit

        # --- Detection: use OpenMM (same as check_chirality in structure_validation.py) ---
        pdb = OMMPDBFile(pdb_path)
        pos = np.asarray(pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64)

        flagged = []  # list of (chain_id, resid_str)
        for res in pdb.topology.residues():
            if res.name == 'GLY':
                continue
            a = {atom.name: atom.index for atom in res.atoms()
                 if atom.name in ('N', 'CA', 'C', 'CB')}
            if len(a) != 4:
                continue
            v_n  = pos[a['N']]  - pos[a['CA']]
            v_c  = pos[a['C']]  - pos[a['CA']]
            v_cb = pos[a['CB']] - pos[a['CA']]
            vol = float(np.dot(np.cross(v_n, v_c), v_cb))
            if vol < 1.0:
                flagged.append((res.chain.id, res.id))

        if not flagged:
            return 0

        # --- Repair: raw PDB line editing with full sidechain reflection ---
        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        flagged_set = set(flagged)
        # Collect all atoms per flagged residue, split backbone vs sidechain
        residue_atoms = {}  # (chain_id, resid) -> {'backbone': {name: xyz}, 'sidechain': [(line_idx, xyz)]}

        for i, line in enumerate(lines):
            if not line.startswith('ATOM'):
                continue
            chain_id = line[21]
            resid = line[22:26].strip()
            key = (chain_id, resid)
            if key not in flagged_set:
                continue

            atom_name = line[12:16].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            xyz = np.array([x, y, z])

            entry = residue_atoms.setdefault(key, {'backbone': {}, 'sidechain': []})
            if atom_name in _CHIRALITY_BACKBONE_ATOMS:
                entry['backbone'][atom_name] = xyz
            else:
                entry['sidechain'].append((i, atom_name, xyz))

        repaired_count = 0
        modified = False

        for (chain_id, resid), data in residue_atoms.items():
            bb = data['backbone']
            if not all(k in bb for k in ('N', 'CA', 'C')):
                continue
            if not data['sidechain']:
                continue

            ca = bb['CA']
            vec_n = bb['N'] - ca
            vec_c = bb['C'] - ca

            # Reflection plane normal (N-CA-C plane)
            normal = np.cross(vec_n, vec_c)
            norm_mag = LA.norm(normal)
            if norm_mag < 1e-6:
                continue
            normal /= norm_mag

            # Reflect every sidechain atom across the plane
            for line_idx, atom_name, atom_pos in data['sidechain']:
                vec_atom = atom_pos - ca
                dist = np.dot(vec_atom, normal)
                new_pos = ca + vec_atom - (2 * dist * normal)

                line = lines[line_idx]
                lines[line_idx] = (
                    line[:30]
                    + f"{new_pos[0]:8.3f}{new_pos[1]:8.3f}{new_pos[2]:8.3f}"
                    + line[54:]
                )

            repaired_count += 1
            modified = True

            if verbose:
                resname = lines[data['sidechain'][0][0]][17:20].strip()
                n_sc = len(data['sidechain'])
                print(f"       [REPAIR] Flipped D-isomer to L-form: "
                      f"{resname} {resid} ({n_sc} sidechain atoms reflected)",
                      flush=True)

        if modified:
            with open(pdb_path, 'w') as f:
                f.writelines(lines)

        return repaired_count

    except Exception as e:
        print(f"       [ERROR] Chirality repair failed: {e}", flush=True)
        return 0


def fix_histidine_naming(pdb_path, residue_ids, verbose=False):
    """
    Fix histidine imidazole ring atom naming after AMBER relaxation.

    AMBER relaxation can intermittently scramble atom identities within the
    HIS imidazole ring, mixing heavy-atom and hydrogen labels.  This function
    reassigns atom names based purely on spatial geometry:

      1. CG identified as the sidechain atom closest to CB (~1.51 A).
      2. Five-membered ring traced via bond-distance adjacency (1.25-1.45 A).
      3. Ring direction (ND1 vs CD2 side) determined by per-atom hydrogen
         count pattern (works for HID, HIE, HIP protonation states).
      4. Hydrogen names reassigned to nearest ring-atom parent.

    Args:
        pdb_path:    Path to the relaxed PDB file (modified in-place).
        residue_ids: Set of residue-ID strings (e.g. {'92', '110'}) for the
                     specific HIS residues to fix (from check_bond_integrity).
        verbose:     Print per-atom rename details.

    Returns:
        Number of residues where naming was actually corrected.
    """
    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    # -- Collect atoms only for the target HIS residues --
    residue_atoms = {}                         # resid -> [(line_idx, name, xyz)]
    for i, line in enumerate(lines):
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            continue
        resname = line[17:20].strip()
        if resname not in _HIS_VARIANTS:
            continue
        resid = line[22:26].strip()
        if resid not in residue_ids:
            continue
        atom_name = line[12:16].strip()
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        residue_atoms.setdefault(resid, []).append(
            (i, atom_name, np.array([x, y, z]))
        )

    if not residue_atoms:
        return 0

    fixed_count = 0
    any_modified = False
    fixed_resids_list = []

    for resid, atoms in residue_atoms.items():
        # -- Separate backbone / sidechain --
        backbone = {}
        sidechain = []
        for line_idx, name, pos in atoms:
            if name in _BACKBONE_NAMES:
                backbone[name] = (line_idx, pos)
            else:
                sidechain.append((line_idx, name, pos))

        if 'CB' not in backbone or len(sidechain) < 5:
            continue

        cb_pos = backbone['CB'][1]

        # ---- A. Find CG: sidechain atom closest to CB ----
        sc_sorted = sorted(sidechain,
                           key=lambda a: float(LA.norm(a[2] - cb_pos)))
        # Build working list:  index 0 = CG candidate, rest follow
        all_sc = list(sc_sorted)
        n = len(all_sc)
        positions = np.array([a[2] for a in all_sc])

        # ---- B. Ring adjacency by bond distance, traverse ring ----
        adj = {i: [] for i in range(n)}
        for a in range(n):
            for b in range(a + 1, n):
                d = float(LA.norm(positions[a] - positions[b]))
                if _RING_BOND_MIN <= d <= _RING_BOND_MAX:
                    adj[a].append(b)
                    adj[b].append(a)

        # CG (index 0) must have exactly 2 ring neighbours
        if len(adj[0]) != 2:
            if verbose:
                print(f"       [WARNING] HIS {resid}: CG has "
                      f"{len(adj[0])} ring neighbours, skipping.",
                      flush=True)
            continue

        arm1, arm2 = adj[0]

        # Walk CG -> arm1 -> ... and check ring closure at arm2
        def _walk(start_arm, close_arm):
            path = [0, start_arm]
            for _ in range(3):
                cur, prev = path[-1], path[-2]
                nxt = [x for x in adj[cur] if x != prev]
                if len(nxt) != 1:
                    return None
                path.append(nxt[0])
            return path if (len(path) == 5 and path[-1] == close_arm) else None

        ring_order = _walk(arm1, arm2) or _walk(arm2, arm1)
        if ring_order is None:
            if verbose:
                print(f"       [WARNING] HIS {resid}: cannot trace "
                      f"5-membered ring, skipping.", flush=True)
            continue

        # ring_order = [CG, X1, X2, X3, X4]  ring: CG-X1-X2-X3-X4-CG
        ring_set = set(ring_order)
        non_ring = [i for i in range(n) if i not in ring_set]

        # ---- C. Count H neighbours per ring atom ----
        h_assign = {}
        for ri in ring_order:
            h_assign[ri] = [ni for ni in non_ring
                            if float(LA.norm(positions[ri] - positions[ni]))
                            <= _H_BOND_MAX]

        pattern = [len(h_assign[ring_order[k]]) for k in range(1, 5)]
        # Canonical forward  CG->ND1->CE1->NE2->CD2:
        #   HID: [1,1,0,1]   HIE: [0,1,1,1]   HIP: [1,1,1,1]

        # ---- D. Determine direction ----
        if   pattern == [1, 1, 0, 1]:  forward = True    # HID fwd
        elif pattern == [1, 0, 1, 1]:  forward = False   # HID rev
        elif pattern == [0, 1, 1, 1]:  forward = True    # HIE fwd
        elif pattern == [1, 1, 1, 0]:  forward = False   # HIE rev
        elif pattern == [1, 1, 1, 1]:                     # HIP
            d1 = float(LA.norm(positions[ring_order[0]]
                               - positions[ring_order[1]]))
            d4 = float(LA.norm(positions[ring_order[0]]
                               - positions[ring_order[4]]))
            forward = (d1 >= d4)       # CG-ND1 (~1.38) > CG-CD2 (~1.35)
        else:
            if verbose:
                print(f"       [WARNING] HIS {resid}: unexpected H-count "
                      f"pattern {pattern}, skipping.", flush=True)
            continue

        if not forward:
            ring_order = [ring_order[0]] + ring_order[1:][::-1]

        # ring_order now maps to [CG, ND1, CE1, NE2, CD2]

        # ---- E. Build name mapping (ring + hydrogens) ----
        name_map = {}
        for k, canon in enumerate(_HIS_RING_NAMES):
            name_map[ring_order[k]] = canon
            h_name = _HIS_H_MAP.get(canon)
            if h_name:
                for ni in h_assign[ring_order[k]]:
                    name_map[ni] = h_name

        # ---- F. Check whether any names actually changed ----
        changes = [(idx, all_sc[idx][1], new)
                   for idx, new in name_map.items()
                   if all_sc[idx][1] != new]

        if not changes:
            continue

        if verbose:
            for _, old, new in changes:
                print(f"       [REPAIR] HIS {resid}: {old} -> {new}",
                      flush=True)

        # ---- G. Rewrite PDB lines ----
        for idx, new_name in name_map.items():
            li = all_sc[idx][0]          # line index in file
            line = lines[li]

            # Atom name  (PDB cols 13-16, 0-indexed 12:16)
            fmt_name = f" {new_name:<3s}" if len(new_name) < 4 \
                       else f"{new_name:<4s}"
            line = line[:12] + fmt_name + line[16:]

            # Element symbol (PDB cols 77-78, 0-indexed 75:78)
            elem = ('H' if new_name[0] == 'H'
                    else 'N' if new_name[0] == 'N'
                    else 'C')
            raw = line.rstrip('\n')
            if len(raw) >= 78:
                raw = raw[:75] + f"{elem:>2s}" + raw[78:]
            else:
                raw = raw.ljust(75) + f"{elem:>2s}"
            lines[li] = raw + '\n'

        any_modified = True
        fixed_count += 1
        fixed_resids_list.append(resid)

    if any_modified:
        print(f"       [REPAIR] Reordered misaligned histidines: "
              f"{', '.join(f'HIS{rid}' for rid in fixed_resids_list)}",
              flush=True)
        with open(pdb_path, 'w') as f:
            f.writelines(lines)

    return fixed_count
