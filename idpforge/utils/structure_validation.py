"""
idpforge/utils/structure_validation.py

Structural validation utilities for protein conformers.

Features:
- CHIRALITY: Checks for D-amino acids using scalar triple product of N, CA, C, CB.
- BONDS: Verifies all covalent bonds are intact (distance < 2.2 Å).
- CLASHES: Backbone-Backbone clash detection with IDR-aware filtering.
- TOPOLOGY: Hybrid Protocol (Topoly Native Implementation).
    1. Alexander Gatekeeper (Probabilistic):
       - Uses native 'tries=100' argument.
       - Threshold 0.65 prevents "borderline" knots from slipping through.
    2. HOMFLY Fallback (Deterministic):
       - Uses Closure.MASS_CENTER to avoid false positives (phantom knots).
"""

from __future__ import annotations
import math
import time
import logging
from typing import Any, Tuple, Dict, List

import numpy as np
from scipy.spatial import cKDTree
from openmm import unit

# -------------------------
# Constants
# -------------------------
VERBOSE = False
VDW_RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80, 'H': 1.20}

# AlphaKnot2 Hybrid Settings
ALEXANDER_TRIES        = 100
# Thresholds for Alexander Gatekeeper
# 0.65 = Must match unknot in 65/100 projections to exit early.
ALPHAKNOT_P_UNKNOT_MAX = 0.65

# ============================================================
# 1. HELPERS
# ============================================================
def _get_coords_64(topology, positions):
    """Safely extracts coordinates as a float64 numpy array."""
    if hasattr(positions, "value_in_unit"):
        pos = positions.value_in_unit(unit.angstrom)
        return np.asarray(pos, dtype=np.float64)
    return np.asarray(positions, dtype=np.float64)

# ============================================================
# 2. VALIDATION LOGIC
# ============================================================
def check_chirality(topology, positions):
    """
    Checks for D-amino acids (chirality violations) in the backbone.
    Returns a list of issues found.
    """
    pos = _get_coords_64(topology, positions)
    issues = []
    for res in topology.residues():
        if res.name == 'GLY': continue

        # We need N, CA, C, CB to check chirality volume
        a = {atom.name: atom.index for atom in res.atoms() if atom.name in ('N', 'CA', 'C', 'CB')}
        if len(a) == 4:
            v_n = pos[a['N']] - pos[a['CA']]
            v_c = pos[a['C']] - pos[a['CA']]
            v_cb = pos[a['CB']] - pos[a['CA']]

            # Scalar triple product
            vol = float(np.dot(np.cross(v_n, v_c), v_cb))

            # Volume < 0 indicates D-enantiomer (should be L)
            if vol < 1.0:
                stereo = "D" if vol < 0 else "Planar/Distorted"
                issues.append({"residue": f"{res.name}{res.id}", "volume": vol, "stereo": stereo})
    return issues

def check_bond_integrity(topology, positions, threshold=2.2):
    """
    Checks for broken covalent bonds (> threshold Angstroms).
    Naive implementation relying entirely on OpenMM's inferred topology.
    """
    positions = _get_coords_64(topology, positions)
    broken = []
    for bond in topology.bonds():
        i, j = bond.atom1.index, bond.atom2.index
        d = float(np.linalg.norm(positions[i] - positions[j]))
        if d > threshold:
            print(f"         [BOND ERROR] {bond.atom1.residue.name}{bond.atom1.residue.id}({bond.atom1.name}) - {bond.atom2.residue.name}{bond.atom2.residue.id}({bond.atom2.name}) d={d:.2f} Å", flush=True)
            broken.append({
                "resname": bond.atom1.residue.name,
                "resid": bond.atom1.residue.id,
                "atom1": bond.atom1.name,  # Added for debugging
                "atom2": bond.atom2.name,  # Added for debugging
                "distance": d,
                "resname2": bond.atom2.residue.name,
                "resid2": bond.atom2.residue.id
            })
    return broken

def check_clashes_detailed(topology, positions, overlap_cutoff=0.4, idr_start=None, idr_end=None):
    """
    Fast Backbone-Backbone Clash Check (cKDTree).

    FILTERS:
    1. Atoms: N, CA, C (Skeleton only, O/OXT removed).
    2. Exclusions:
       - Bonded (i to i+1).
       - Folded-Folded interactions (Ignored if idr_start/end provided).

    Returns: (score, count, details)
    """
    # 1. Minimal Backbone (Skeleton Only)
    BACKBONE_NAMES = {'N', 'CA', 'C'}

    atoms = list(topology.atoms())

    # Storage arrays
    bb_indices = []
    bb_radii = []
    bb_res_ids = []
    bb_chain_ids = []

    # Pre-fetch radii (using simple dictionary lookup)
    # Note: Carbon (C) radius ~1.70A, Nitrogen (N) ~1.55A
    def get_r(elem): return VDW_RADII.get(elem, 1.70)

    for i, atom in enumerate(atoms):
        if atom.name in BACKBONE_NAMES:
            bb_indices.append(i)
            bb_radii.append(get_r(atom.element.symbol))
            bb_res_ids.append(int(atom.residue.id))
            bb_chain_ids.append(atom.residue.chain.index)

    if not bb_indices: return 0.0, 0, []

    # 2. Build Tree
    all_coords = _get_coords_64(topology, positions)
    coords = all_coords[bb_indices]
    radii = np.array(bb_radii)
    res_ids = np.array(bb_res_ids)
    chain_ids = np.array(bb_chain_ids)

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=4.0)

    clash_count = 0

    # 3. Check Pairs with IDR Logic
    for i, j in pairs:
        # A. Chain/Bond Exclusion
        # Ignore if Same Chain AND (Same Residue OR Adjacent Residue)
        if chain_ids[i] == chain_ids[j]:
            if abs(res_ids[i] - res_ids[j]) <= 1:
                continue

        # B. Domain Logic (IDR vs Folded)
        # If IDR range is defined, we skip checks where BOTH atoms are Folded.
        if idr_start is not None and idr_end is not None:
            # Check if Atom I is in IDR
            is_i_idr = (idr_start <= res_ids[i] <= idr_end)
            # Check if Atom J is in IDR
            is_j_idr = (idr_start <= res_ids[j] <= idr_end)

            # If NEITHER is in the IDR (meaning both are folded), SKIP.
            if not is_i_idr and not is_j_idr:
                continue

        # C. Overlap Calculation
        dist = np.linalg.norm(coords[i] - coords[j])
        overlap = (radii[i] + radii[j]) - dist

        # We are counting ERRORS, so Overlap >= cutoff is BAD.
        if overlap >= overlap_cutoff:
            clash_count += 1

    score = (clash_count / len(bb_indices)) * 1000.0
    return score, clash_count, []

# ============================================================
# 3. TOPOLOGY (AlphaKnot2 Hybrid)
# ============================================================
def classify_global_topology_alphaknot(topology, positions, VERBOSE=False):
    """
    Determines if protein is knotted using a Hybrid approach:
    1. Alexander Polynomial (Probabilistic, Fast):
       - Calls tp.alexander with tries=100.
       - Returns dictionary of probabilities.
    2. HOMFLY Polynomial (Deterministic, Slow):
       - Used if Alexander confidence < 0.65.
       - Uses Closure.MASS_CENTER.
    """
    import topoly as tp
    from topoly.params import Closure

    # Prepare Coordinates (CA only)
    pos = _get_coords_64(topology, positions)
    ca_atoms = sorted([a for a in topology.atoms() if a.name == "CA"], key=lambda x: int(x.residue.id))
    if not ca_atoms: return {"label": "None", "reason": "NoCA"}

    coords_list = pos[[a.index for a in ca_atoms]].tolist()

    # ----------------------------------------
    # Phase 1: Alexander Probabilistic Gatekeeper
    # ----------------------------------------
    try:
        # Native 'tries' argument handles the loop internally (much faster)
        # Returns dict, e.g., {'0_1': 0.95, '3_1': 0.05}
        alex_results = tp.alexander(coords_list, tries=ALEXANDER_TRIES)
    except Exception:
        alex_results = {}

    # Calculate Probability of Unknot
    # Sum up all keys that represent unknot
    alex_p_unknot = 0.0
    if isinstance(alex_results, dict):
        for k, v in alex_results.items():
            if str(k) in ["0_1", "Unknot", "0", "1"]:
                alex_p_unknot += v

    alex_p_knot = 1.0 - alex_p_unknot

    # ----------------------------------------
    # Phase 2: Decision Logic
    # ----------------------------------------

    # CASE A: High Confidence Unknot
    if alex_p_unknot >= ALPHAKNOT_P_UNKNOT_MAX:
        return {
            "label": "None",
            "closure_polys": [f"Alex_P_Unknot={alex_p_unknot:.2f}"],
            "Alexander_Ran": True,
            "HOMFLY_Ran": False,
            "reason": f"HighConf_Unknot={alex_p_unknot:.2f}"
        }

    # CASE B: Ambiguous (Gray Zone) OR Likely Knot -> Run HOMFLY
    # Use Deterministic MASS_CENTER closure to prevent phantom knots
    try:
        # No 'tries' needed for deterministic closure
        poly = tp.homfly(coords_list, closure=Closure.MASS_CENTER)

        if poly is None:
            return {"label": "Error", "reason": "HOMFLY_Failed", "Alexander_Ran": True, "HOMFLY_Ran": True}

        # Check for Unknot in HOMFLY output
        is_unknot_homfly = False
        if isinstance(poly, str) and poly in ["0_1", "Unknot", "0", "1"]:
            is_unknot_homfly = True
        elif isinstance(poly, dict):
            # Should not happen with deterministic closure, but safety first
            is_unknot_homfly = "0_1" in poly or "Unknot" in poly

        if is_unknot_homfly:
             return {
                "label": "None",
                "closure_polys": [f"Alex_P={alex_p_knot:.2f}|HOMFLY={str(poly)}"],
                "Alexander_Ran": True,
                "HOMFLY_Ran": True,
                "reason": "HOMFLY_Unknot"
            }
        else:
             return {
                "label": "Knot",
                "closure_polys": [f"Alex_P={alex_p_knot:.2f}|HOMFLY={str(poly)}"],
                "Alexander_Ran": True,
                "HOMFLY_Ran": True,
                "reason": f"HOMFLY_Knot({str(poly)})"
            }

    except Exception as e:
        return {"label": "Error", "reason": f"HOMFLY_Ex({str(e)})", "Alexander_Ran": True, "HOMFLY_Ran": True}

# ============================================================
# 4. MAIN ENTRY POINT
# ============================================================
def validate_structure_post_relax(
    topology, positions, pdb_path="", strict_clash_threshold=10.0,
    idr_start=None, idr_end=None, attempts=None, verbose=False,
    full_report=False
):
    """
    Main validation function.

    Args:
        full_report: If True, runs ALL checks regardless of failures and
                     populates every field in the info dict. If False
                     (default), short-circuits on the first failure for speed.
    """
    info = {"pdb_path": pdb_path}
    all_pass = True

    # 1. Chirality
    t0 = time.perf_counter()
    flips = check_chirality(topology, positions)
    info["Time_Chirality_s"] = round(time.perf_counter() - t0, 6)
    info["chirality_pass"] = (len(flips) == 0)
    if not info["chirality_pass"]:
        all_pass = False
        info["chirality_detail"] = f"{len(flips)} issues"
        info["chirality_error_residue"] = flips[0]["residue"]
        info["chirality_error_stereo"] = flips[0]["stereo"]
        info["chirality_error_volume"] = flips[0]["volume"]
        if not full_report:
            info["reason"] = "Chirality"
            return False, info

    # 2. Bonds
    t0 = time.perf_counter()
    broken = check_bond_integrity(topology, positions, threshold=2.2)
    info["Time_Bonds_s"] = round(time.perf_counter() - t0, 6)
    info["bonds_pass"] = (len(broken) == 0)
    info["num_broken_bonds"] = len(broken)
    if not info["bonds_pass"]:
        all_pass = False
        first = broken[0]
        if verbose:
            print(f"         [DEBUG] Broken bond: {first['resname']}{first['resid']} "
                  f"({first['atom1']}-{first['atom2']}) d={first['distance']:.2f} Å")
        info["broken_bonds_first_res"] = f"{first['resname']}{first['resid']}"
        if not full_report:
            info["reason"] = "BrokenBonds"
            return False, info

    # 3. Clashes (Backbone)
    t0 = time.perf_counter()
    score, n_clashes, _ = check_clashes_detailed(
        topology, positions, idr_start=idr_start, idr_end=idr_end
    )
    info["Time_Clashes_s"] = round(time.perf_counter() - t0, 6)
    info["clash_score"] = score
    info["num_clashes"] = n_clashes
    info["clash_pass"] = (score <= strict_clash_threshold)
    if not info["clash_pass"]:
        all_pass = False
        if not full_report:
            info["reason"] = "Clashscore"
            return False, info

    # 4. Topology (Hybrid: Alexander -> HOMFLY)
    t0 = time.perf_counter()
    topo = classify_global_topology_alphaknot(topology, positions, VERBOSE=verbose)
    info["Time_Knots_s"] = round(time.perf_counter() - t0, 6)
    info["knot_pass"] = (topo["label"] == "None")
    info["knot_type"] = topo.get("label")
    info["Topology_Source"] = "AlexanderProb" if not topo.get("HOMFLY_Ran") else "HOMFLY"
    info["Alexander_Ran"] = topo.get("Alexander_Ran", False)
    info["HOMFLY_Ran"] = topo.get("HOMFLY_Ran", False)
    if not info["knot_pass"]:
        all_pass = False
        if not full_report:
            info["reason"] = topo.get("reason", "Knot")
            return False, info

    # Build composite reason
    if all_pass:
        info["reason"] = "OK"
    else:
        reasons = []
        if not info["chirality_pass"]:
            reasons.append("Chirality")
        if not info["bonds_pass"]:
            reasons.append("BrokenBonds")
        if not info["clash_pass"]:
            reasons.append("Clashscore")
        if not info["knot_pass"]:
            reasons.append(topo.get("reason", "Knot"))
        info["reason"] = ", ".join(reasons)

    return all_pass, info
