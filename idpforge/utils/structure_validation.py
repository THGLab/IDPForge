"""Structural validation utilities for protein conformers."""

from __future__ import annotations
import json
import math
import os
import re
import time
import logging
from typing import Any, Tuple, Dict, List, Optional

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
ALPHAKNOT_P_UNKNOT_MAX = 0.65

_KNOT_RE = re.compile(r"HOMFLY_Knot\((.+)\)")


# -------------------------
# Curvature-gate window
# -------------------------
_JUNCTION_KAPPA_WINDOW = 20


# ============================================================
# 0. KNOT SCREENING UTILITIES
# ============================================================
def extract_knot_type(reason: Optional[str]) -> Optional[str]:
    """Extract the canonical knot type from a topology reason string."""
    if reason is None:
        return None
    m = _KNOT_RE.search(reason)
    return m.group(1) if m else None


def load_knot_screening(json_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Return {protein_id: [{"range": [start, end], "knot": K_or_None}, ...]}."""
    if not os.path.isfile(json_path):
        return {}
    with open(json_path) as f:
        raw = json.load(f)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for pid, info in raw.items():
        if not isinstance(info, dict):
            continue
        domains = info.get("domains")
        if isinstance(domains, list):
            spec: List[Dict[str, Any]] = []
            for d in domains:
                if not isinstance(d, dict):
                    continue
                rng = d.get("range")
                if not (isinstance(rng, list) and len(rng) == 2):
                    continue
                spec.append({"range": [int(rng[0]), int(rng[1])],
                             "knot": d.get("knot")})
            out[pid] = spec
    return out


def format_domain_spec(spec: Optional[List[Dict[str, Any]]]) -> str:
    """Human-readable summary of an expected per-domain knot spec."""
    if spec is None:
        return "None"
    if not spec:
        return "no-folded"
    parts = []
    for i, d in enumerate(spec, start=1):
        rng = d.get("range", [None, None])
        knot = d.get("knot")
        parts.append(f"F{i}[{rng[0]}-{rng[1]}]={knot or 'U'}")
    return ",".join(parts)


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
    """Check for D-amino acids (chirality violations) in the backbone."""
    pos = _get_coords_64(topology, positions)
    issues = []
    for res in topology.residues():
        if res.name == 'GLY': continue

        # N, CA, C, CB for the chirality volume
        a = {atom.name: atom.index for atom in res.atoms() if atom.name in ('N', 'CA', 'C', 'CB')}
        if len(a) == 4:
            v_n = pos[a['N']] - pos[a['CA']]
            v_c = pos[a['C']] - pos[a['CA']]
            v_cb = pos[a['CB']] - pos[a['CA']]

            # Scalar triple product
            vol = float(np.dot(np.cross(v_n, v_c), v_cb))

            if vol < 1.0:
                stereo = "D" if vol < 0 else "Planar/Distorted"
                issues.append({"residue": f"{res.name}{res.id}", "volume": vol, "stereo": stereo})
    return issues

# Max heavy-atom covalent bond length
_COVALENT_CUTOFF = 2.0


def _bond_atom_is_heavy(atom):
    sym = atom.element.symbol if getattr(atom, "element", None) is not None else atom.name.strip()[:1]
    return (sym or "").upper() != "H"


def check_bond_integrity(topology, positions, threshold=2.2, covalent_cutoff=_COVALENT_CUTOFF):
    """Geometry/graph-based bond-integrity check (atom-name agnostic)."""
    from collections import defaultdict
    positions = _get_coords_64(topology, positions)
    atoms = list(topology.atoms())

    res_heavy = defaultdict(list)              # residue.index -> [heavy atom global indices]
    exp_deg = defaultdict(lambda: defaultdict(int))   # ridx -> {atom_index: expected intra heavy degree}
    exp_edges = defaultdict(list)              # ridx -> [(atom1, atom2)] expected intra heavy bonds
    inter = []                                  # heavy inter-residue bonds (atom1, atom2)
    for a in atoms:
        if _bond_atom_is_heavy(a):
            res_heavy[a.residue.index].append(a.index)
    for bond in topology.bonds():
        a1, a2 = bond.atom1, bond.atom2
        if not (_bond_atom_is_heavy(a1) and _bond_atom_is_heavy(a2)):
            continue
        if a1.residue.index == a2.residue.index:
            exp_deg[a1.residue.index][a1.index] += 1
            exp_deg[a1.residue.index][a2.index] += 1
            exp_edges[a1.residue.index].append((a1, a2))
        else:
            inter.append((a1, a2))

    broken = []
    # --- intra-residue: heavy-atom graph comparison ---
    for ridx, idxs in res_heavy.items():
        edeg = exp_deg.get(ridx)
        if not edeg:
            continue
        exp_seq = sorted(edeg.get(i, 0) for i in idxs)
        obs = {i: 0 for i in idxs}
        n = len(idxs)
        for ii in range(n):
            pi = positions[idxs[ii]]
            for jj in range(ii + 1, n):
                if float(np.linalg.norm(pi - positions[idxs[jj]])) <= covalent_cutoff:
                    obs[idxs[ii]] += 1
                    obs[idxs[jj]] += 1
        obs_seq = sorted(obs.values())
        if obs_seq == exp_seq:
            continue
        res = atoms[idxs[0]].residue
        missing = (sum(exp_seq) - sum(obs_seq)) // 2          # >0 => fewer bonds than canonical
        disconnected = (min(obs_seq) == 0 and min(exp_seq) >= 1)
        if missing > 0 or disconnected:
            # Broken heavy-atom bond
            stretched = []
            for b1, b2 in exp_edges.get(ridx, []):
                d = float(np.linalg.norm(positions[b1.index] - positions[b2.index]))
                if d > covalent_cutoff:
                    stretched.append((b1.name, b2.name, d))
            stretched.sort(key=lambda t: -t[2])
            tag = "HIS" if res.name in ("HIS", "HID", "HIE", "HIP") else res.name
            if stretched:
                detail = "; ".join(f"{a}-{b} = {d:.2f} A" for a, b, d in stretched[:4])
                print(f"         [BOND] {tag}{res.id} broken (intra: {detail} > {covalent_cutoff})", flush=True)
            else:
                print(f"         [BOND] {tag}{res.id} broken (intra: heavy-degree {obs_seq} != expected {exp_seq})", flush=True)
            broken.append({
                "resname": res.name, "resid": res.id,
                "atom1": stretched[0][0] if stretched else "(residue graph)",
                "atom2": stretched[0][1] if stretched else "(missing bond)",
                "distance": stretched[0][2] if stretched else float("nan"),
                "resname2": res.name, "resid2": res.id,
            })
    # --- inter-residue bonds: direct distance ---
    for a1, a2 in inter:
        d = float(np.linalg.norm(positions[a1.index] - positions[a2.index]))
        if d > threshold:
            print(f"         [BOND] {a1.residue.name}{a1.residue.id}-{a2.residue.name}{a2.residue.id} broken "
                  f"(inter: {a1.name}-{a2.name} = {d:.2f} A > {threshold})", flush=True)
            broken.append({
                "resname": a1.residue.name, "resid": a1.residue.id,
                "atom1": a1.name, "atom2": a2.name, "distance": d,
                "resname2": a2.residue.name, "resid2": a2.residue.id,
            })
    return broken

def check_clashes_detailed(topology, positions, overlap_cutoff=0.4, idr_start=None, idr_end=None):
    """Fast backbone-backbone clash check (cKDTree)."""
    # 1. Minimal Backbone (Skeleton Only)
    BACKBONE_NAMES = {'N', 'CA', 'C'}

    atoms = list(topology.atoms())

    # Storage arrays
    bb_indices = []
    bb_radii = []
    bb_res_ids = []
    bb_chain_ids = []

    # Pre-fetch radii
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
        if chain_ids[i] == chain_ids[j]:
            if abs(res_ids[i] - res_ids[j]) <= 1:
                continue

        # B. Domain Logic (IDR vs Folded)
        if idr_start is not None and idr_end is not None:
            is_i_idr = (idr_start <= res_ids[i] <= idr_end)
            is_j_idr = (idr_start <= res_ids[j] <= idr_end)

            if not is_i_idr and not is_j_idr:
                continue

        # C. Overlap Calculation
        dist = np.linalg.norm(coords[i] - coords[j])
        overlap = (radii[i] + radii[j]) - dist

        if overlap >= overlap_cutoff:
            clash_count += 1

    score = (clash_count / len(bb_indices)) * 1000.0
    return score, clash_count, []

# ============================================================
# 3. TOPOLOGY (AlphaKnot2 Hybrid)
# ============================================================
def classify_global_topology_alphaknot(topology, positions, VERBOSE=False):
    """Determine if the protein is knotted (Alexander -> HOMFLY hybrid)."""
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
        alex_results = tp.alexander(coords_list, tries=ALEXANDER_TRIES)
    except Exception:
        alex_results = {}

    # Calculate Probability of Unknot
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
    try:
        poly = tp.homfly(coords_list, closure=Closure.MASS_CENTER)

        if poly is None:
            return {"label": "Error", "reason": "HOMFLY_Failed", "Alexander_Ran": True, "HOMFLY_Ran": True}

        # Check for Unknot in HOMFLY output
        is_unknot_homfly = False
        if isinstance(poly, str) and poly in ["0_1", "Unknot", "0", "1"]:
            is_unknot_homfly = True
        elif isinstance(poly, dict):
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


def _classify_coords_alphaknot(coords_list):
    """Run the Alexander/HOMFLY hybrid classifier on a CA coordinate list."""
    import topoly as tp
    from topoly.params import Closure

    if not coords_list:
        return {"label": "None", "reason": "NoCA",
                "Alexander_Ran": False, "HOMFLY_Ran": False}

    try:
        alex_results = tp.alexander(coords_list, tries=ALEXANDER_TRIES)
    except Exception:
        alex_results = {}

    alex_p_unknot = 0.0
    if isinstance(alex_results, dict):
        for k, v in alex_results.items():
            if str(k) in ["0_1", "Unknot", "0", "1"]:
                alex_p_unknot += v
    alex_p_knot = 1.0 - alex_p_unknot

    if alex_p_unknot >= ALPHAKNOT_P_UNKNOT_MAX:
        return {
            "label": "None",
            "closure_polys": [f"Alex_P_Unknot={alex_p_unknot:.2f}"],
            "Alexander_Ran": True,
            "HOMFLY_Ran": False,
            "reason": f"HighConf_Unknot={alex_p_unknot:.2f}"
        }

    try:
        poly = tp.homfly(coords_list, closure=Closure.MASS_CENTER)
        if poly is None:
            return {"label": "Error", "reason": "HOMFLY_Failed",
                    "Alexander_Ran": True, "HOMFLY_Ran": True}

        is_unknot_homfly = False
        if isinstance(poly, str) and poly in ["0_1", "Unknot", "0", "1"]:
            is_unknot_homfly = True
        elif isinstance(poly, dict):
            is_unknot_homfly = "0_1" in poly or "Unknot" in poly

        if is_unknot_homfly:
            return {
                "label": "None",
                "closure_polys": [f"Alex_P={alex_p_knot:.2f}|HOMFLY={str(poly)}"],
                "Alexander_Ran": True,
                "HOMFLY_Ran": True,
                "reason": "HOMFLY_Unknot"
            }
        return {
            "label": "Knot",
            "closure_polys": [f"Alex_P={alex_p_knot:.2f}|HOMFLY={str(poly)}"],
            "Alexander_Ran": True,
            "HOMFLY_Ran": True,
            "reason": f"HOMFLY_Knot({str(poly)})"
        }

    except Exception as e:
        return {"label": "Error", "reason": f"HOMFLY_Ex({str(e)})",
                "Alexander_Ran": True, "HOMFLY_Ran": True}


def classify_per_domain_topology(topology, positions, domains):
    """Classify each folded domain range independently."""
    pos = _get_coords_64(topology, positions)
    ca_by_resid: Dict[int, Any] = {}
    for a in topology.atoms():
        if a.name != "CA":
            continue
        try:
            rid = int(a.residue.id)
        except (TypeError, ValueError):
            continue
        if rid not in ca_by_resid:
            ca_by_resid[rid] = pos[a.index].tolist()

    out = []
    for d in domains:
        start, end = d["range"][0], d["range"][1]
        expected_knot = d.get("knot")
        coords = [ca_by_resid[r] for r in range(start, end + 1) if r in ca_by_resid]
        if not coords:
            out.append({
                "range": [start, end],
                "label": "Error",
                "knot": None,
                "reason": "NoCA_in_range",
                "expected_knot": expected_knot,
            })
            continue
        cls = _classify_coords_alphaknot(coords)
        detected = extract_knot_type(cls.get("reason")) if cls.get("label") == "Knot" else None
        out.append({
            "range": [start, end],
            "label": cls.get("label", "Error"),
            "knot": detected,
            "reason": cls.get("reason", ""),
            "expected_knot": expected_knot,
        })
    return out


# ============================================================
# 4. POST-RELAX FOLD GATES
# ============================================================
def _check_fold_gate(atom37, folded_mask, ref_ca, ref_mask, threshold,
                     inclusion_radius=15.0, thresholds=(0.5, 1.0, 2.0, 4.0)):
    """Superposition-free CA-lDDT quality gate over the folded region."""
    folded_mask = np.asarray(folded_mask, dtype=bool)
    sel = folded_mask if ref_mask is None else (folded_mask & np.asarray(ref_mask, dtype=bool))
    model_ca = np.asarray(atom37)[:, 1, :][sel]
    rca = np.asarray(ref_ca)[sel]

    # Drop all-zero (missing/padding) rows
    valid = (np.abs(model_ca).sum(-1) > 1e-6) & (np.abs(rca).sum(-1) > 1e-6)
    model_ca, rca = model_ca[valid], rca[valid]
    n = model_ca.shape[0]
    if n < 2:
        return True, [f"folded CA-lDDT skipped (n_folded={n})"], ""

    Dm = np.linalg.norm(model_ca[:, None, :] - model_ca[None, :, :], axis=-1)
    Dr = np.linalg.norm(rca[:, None, :] - rca[None, :, :], axis=-1)
    included = (Dr < inclusion_radius) & ~np.eye(n, dtype=bool)
    n_pairs = int(included.sum())
    if n_pairs == 0:
        return True, [f"folded CA-lDDT skipped (no pairs < {inclusion_radius} A)"], ""

    diff = np.abs(Dm - Dr)[included]
    score = float(np.mean([(diff < t).mean() for t in thresholds]))
    passes = score >= threshold
    lines = [f"folded CA-lDDT = {score:.3f} (n_folded={n}, pairs={n_pairs}, thr={threshold:.2f})"]
    reason = "" if passes else f"folded-lDDT {score:.3f} < {threshold:.2f}"
    return passes, lines, reason


def _curvature_per_residue(ca, ca_ca_max=4.5):
    """Discrete backbone curvature per residue (kappa_i = 2*sin(theta_i/2) / |r_{i+1}-r_{i-1}|)."""
    ca = np.asarray(ca, dtype=float)
    N = len(ca)
    kappa = np.full(N, np.nan)
    if N < 3:
        return kappa
    brk = np.linalg.norm(np.diff(ca, axis=0), axis=1) > ca_ca_max
    for i in range(1, N - 1):
        if brk[i - 1] or brk[i]:
            continue
        v1 = ca[i] - ca[i - 1]
        v2 = ca[i + 1] - ca[i]
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        chord = np.linalg.norm(ca[i + 1] - ca[i - 1])
        if n1 == 0.0 or n2 == 0.0 or chord == 0.0:
            continue
        cos_th = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        kappa[i] = 2.0 * np.sin(np.arccos(cos_th) / 2.0) / chord
    return kappa


def _check_junction_curvature(ca_coords, viol_mask, kappa_min,
                              window=_JUNCTION_KAPPA_WINDOW):
    """Reject conformers whose IDR was pulled taut to reach the fold anchor."""
    kappa = _curvature_per_residue(ca_coords)
    N = len(viol_mask)
    details = []
    for i in range(N - 1):
        if viol_mask[i] == viol_mask[i + 1]:
            continue
        if viol_mask[i]:
            idxs = [j for j in range(i, max(-1, i - window), -1) if viol_mask[j]]
        else:
            idxs = [j for j in range(i + 1, min(N, i + 1 + window)) if viol_mask[j]]
        kw = kappa[idxs]
        kw = kw[np.isfinite(kw)]
        if len(kw) == 0:
            continue
        mk = float(kw.mean())
        details.append({"idr_lo": min(idxs), "idr_hi": max(idxs),
                        "mean_kappa": mk, "pass": mk >= kappa_min})
    return (all(d["pass"] for d in details) if details else True), details


def _check_fold_curvature(model_ca, ref_ca, folded_mask, min_ratio,
                          window=15, min_ref_kappa=0.03):
    """Reject conformers whose junction-adjacent folded residues were pulled into a straight line."""
    model_ca = np.asarray(model_ca, dtype=float)
    ref_ca = np.asarray(ref_ca, dtype=float)
    folded = np.asarray(folded_mask, dtype=bool)
    N = len(folded)
    k_mod = _curvature_per_residue(model_ca)
    k_ref = _curvature_per_residue(ref_ca)

    details = []
    for i in range(N - 1):
        if folded[i] == folded[i + 1]:
            continue
        if folded[i]:
            idxs = [j for j in range(i, max(-1, i - window), -1) if folded[j]]
        else:
            idxs = [j for j in range(i + 1, min(N, i + 1 + window)) if folded[j]]
        idxs = np.asarray(idxs, dtype=int)
        both = np.isfinite(k_mod[idxs]) & np.isfinite(k_ref[idxs])
        if int(both.sum()) < 3:
            continue
        km = float(np.mean(k_mod[idxs][both]))
        kr = float(np.mean(k_ref[idxs][both]))
        applied = kr >= min_ref_kappa
        passes_j = (km >= min_ratio * kr) if applied else True
        details.append({"fold_lo": int(idxs.min()), "fold_hi": int(idxs.max()),
                        "k_model": km, "k_ref": kr, "applied": applied, "pass": passes_j})

    passes = all(d["pass"] for d in details) if details else True
    applied = [d for d in details if d["applied"]]
    if applied:
        worst = min(applied, key=lambda d: (d["k_model"] / d["k_ref"]) if d["k_ref"] else 1.0)
        frac = worst["k_model"] / worst["k_ref"] if worst["k_ref"] else float("nan")
        lines = [f"fold curvature model/ref = {frac:.2f} "
                 f"(k_model={worst['k_model']:.3f}, k_ref={worst['k_ref']:.3f}, "
                 f"thr={min_ratio:.2f}, win={window})"]
        reason = "" if passes else (
            f"fold-straightened model/ref={frac:.2f} < {min_ratio:.2f} "
            f"(k_model={worst['k_model']:.3f} vs ref {worst['k_ref']:.3f})")
    else:
        lines = [f"fold curvature: no curved junction window (win={window})"]
        reason = ""
    return passes, lines, reason



# ============================================================
# 5. MAIN ENTRY POINT
# ============================================================
def validate_structure_post_relax(
    topology, positions, pdb_path="", strict_clash_threshold=10.0,
    idr_start=None, idr_end=None, attempts=None, verbose=False,
    full_report=False, expected_knot_type=None
):
    """Main validation function."""
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
    info["expected_knot_type"] = expected_knot_type
    info["expected_knot_display"] = format_domain_spec(expected_knot_type) \
        if isinstance(expected_knot_type, list) else str(expected_knot_type)

    if isinstance(expected_knot_type, list):
        if not expected_knot_type:
            info["Time_Knots_s"] = round(time.perf_counter() - t0, 6)
            info["Topology_Source"] = "PerDomain(empty)"
            info["Alexander_Ran"] = False
            info["HOMFLY_Ran"] = False
            info["knot_type"] = "None"
            info["detected_knot_type"] = None
            info["domain_topology"] = []
            info["knot_pass"] = True
        else:
            per_dom = classify_per_domain_topology(topology, positions, expected_knot_type)
            info["Time_Knots_s"] = round(time.perf_counter() - t0, 6)
            info["domain_topology"] = per_dom
            info["Alexander_Ran"] = any(d.get("label") != "Error" for d in per_dom)
            info["HOMFLY_Ran"] = any("HOMFLY" in d.get("reason", "") for d in per_dom)
            info["Topology_Source"] = "PerDomain"

            mismatches = []
            errors = []
            for i, d in enumerate(per_dom, start=1):
                if d["label"] == "Error":
                    errors.append(f"F{i}[{d['range'][0]}-{d['range'][1]}]:{d.get('reason', '')}")
                    continue
                if d["knot"] != d["expected_knot"]:
                    mismatches.append(
                        f"F{i}[{d['range'][0]}-{d['range'][1]}]:"
                        f"exp={d['expected_knot'] or 'U'},got={d['knot'] or 'U'}"
                    )

            info["knot_type"] = "Knot" if any(d["knot"] for d in per_dom) else "None"
            info["detected_knot_type"] = ",".join(
                f"F{i}={d['knot'] or 'U'}" for i, d in enumerate(per_dom, start=1)
            )
            info["knot_pass"] = (not mismatches) and (not errors)
            if not info["knot_pass"]:
                reason_bits = []
                if mismatches:
                    reason_bits.append("KnotMismatch(" + "|".join(mismatches) + ")")
                if errors:
                    reason_bits.append("KnotError(" + "|".join(errors) + ")")
                info["knot_fail_reason"] = ";".join(reason_bits)
    else:
        topo = classify_global_topology_alphaknot(topology, positions, VERBOSE=verbose)
        info["Time_Knots_s"] = round(time.perf_counter() - t0, 6)
        info["Topology_Source"] = "AlexanderProb" if not topo.get("HOMFLY_Ran") else "HOMFLY"
        info["Alexander_Ran"] = topo.get("Alexander_Ran", False)
        info["HOMFLY_Ran"] = topo.get("HOMFLY_Ran", False)

        detected_knot = extract_knot_type(topo.get("reason"))
        info["knot_type"] = topo.get("label")
        info["detected_knot_type"] = detected_knot

        if expected_knot_type is not None:
            info["knot_pass"] = (detected_knot == expected_knot_type)
        else:
            info["knot_pass"] = (topo["label"] == "None")

        if not info["knot_pass"]:
            info["knot_fail_reason"] = topo.get("reason", "Knot")

    if not info["knot_pass"]:
        all_pass = False
        if not full_report:
            info["reason"] = info.get("knot_fail_reason", "Knot")
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
            reasons.append(info.get("knot_fail_reason", "Knot"))
        info["reason"] = ", ".join(reasons)

    return all_pass, info
