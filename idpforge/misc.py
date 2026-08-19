"""
Modified from ESM2
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import typing as T
import gc
import time
import torch
import numpy as np
import os
import re
import uuid

from esm.esmfold.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
)

from openfold.np.protein import Protein as OFProtein
from openfold.np.protein import to_pdb, from_pdb_string
from openfold.utils.feats import atom14_to_atom37

from idpforge.utils.definitions import (
    sstype_order, ss_lib, sstype_num, 
    coil_types, coil_sample_probs,
)
from idpforge.utils.np_utils import calc_rg, assign_rama
            


def _ca_from_pdb_file(path, n_res):
    """Parse CA coordinates from a PDB into an (n_res, 3) array."""
    ca = np.zeros((n_res, 3), dtype=float)
    seen = []
    with open(path) as fh:
        for line in fh:
            if line[:6].strip() in ("ATOM", "HETATM") and line[12:16].strip() == "CA":
                try:
                    resseq = int(line[22:26])
                    xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                except ValueError:
                    continue
                seen.append((resseq, xyz))
    for pos, (_, xyz) in enumerate(sorted(seen, key=lambda t: t[0])):
        if pos >= n_res:
            break
        ca[pos] = xyz
    return ca


def onehot_to_ss(tokens, mask):
    return "".join([ss_lib[t] for t in tokens[mask]])
    
def encode_ss(
    ss: str,
    backbone_tor: T.Optional[torch.Tensor] = None,
    sample_coil: bool = False,
    chain_linker: T.Optional[str] = "C" * 25) -> torch.Tensor:

    if backbone_tor is not None:
        rama = assign_rama(torch.atan2(backbone_tor[..., 0], backbone_tor[..., 1]).numpy())
        ss = "".join([d if d in ["H", "E"] else r for d, r in zip(ss, rama)])
    elif sample_coil:
        ss = "".join([np.random.choice(list(coil_sample_probs.keys()), 
                                       p=list(coil_sample_probs.values())) if i in coil_types \
                      else i for i in ss])
    ss = re.sub(r'[AH]{6,}', lambda match: 'H' * len(match.group()), ss)
    if chain_linker is None:
        chain_linker = ""
    chains = ss.split(":")
    seq = chain_linker.join(chains)
    return torch.tensor([sstype_order.get(s) for s in seq])
    
def batch_encode_ss(
    ss: T.Sequence[str],
    backbone_tor: T.Optional[T.Sequence[torch.Tensor]] = None,
    chain_linker: T.Optional[str] = "C" * 25,
) -> T.Tuple[torch.Tensor, torch.Tensor]:

    sstype_list = []
    if backbone_tor is None:
        backbone_tor = [None] * len(ss) 
    for s, tor in zip(ss, backbone_tor):
        sstype_seq = encode_ss(
            s, tor, # (N, 2, 2)
            s.count("B")+s.count("A")==0,
            chain_linker=chain_linker,
        )
        sstype_list.append(sstype_seq)

    sstype = collate_dense_tensors(sstype_list)
    mask = collate_dense_tensors(
        [sstype.new_ones(len(sstype_seq)) for sstype_seq in sstype_list]
    )

    return sstype, mask

def input_process(
    sequences: T.Sequence[str], 
    ss: T.Sequence[str],
    backbone_tor: T.Optional[T.Sequence[torch.Tensor]] = None,
    residx = None,
    residue_index_offset: T.Optional[int] = 512, 
    chain_linker: T.Optional[str] = "G" * 25):
    aatype, aa_mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            sequences, residue_index_offset, chain_linker
        )
    sstype, ss_mask = batch_encode_ss(ss, backbone_tor, chain_linker.replace("G", "C"))
    assert torch.equal(aa_mask, ss_mask)

    if residx is None:
        residx = _residx
    elif not isinstance(residx, torch.Tensor):
        residx = collate_dense_tensors(residx)
    return aatype, sstype, aa_mask, residx, linker_mask


def output_to_pdb(
    output: dict,
    select_idx=None,
    relax=None,
    save_path=None,
    counter=1,
    counter_cap=None,
    verbose=False,
    fold_ref_ca=None,
    fold_mask=None,
    fold_curv_ratio=None,
    fold_curv_window=15,
    junction_kappa=0.0,
    expected_knot_type=None,
    orig_resid=None,
    **kwargs
):
    """Writes ATOM37 PDBs (with hydrogens) using numbered filenames."""

    # Convert atom14 → atom37
    final_atom_positions = atom14_to_atom37(output["positions"], output)
    final_atom_positions = final_atom_positions.numpy()

    # Convert all tensors to numpy
    output = {k: v.numpy() for k, v in output.items()}
    final_atom_mask = output["atom37_atom_exists"].astype(bool)

    if select_idx is None:
        select_idx = range(output["aatype"].shape[0])

    # IDR mask for the pre-relax junction filter
    viol_mask = kwargs.get("viol_mask")

    # Map a local position to the displayed residue number
    def _disp_resid(pos):
        if orig_resid is not None and 0 <= pos < len(orig_resid):
            return int(orig_resid[pos]) + 1
        return pos + 1

    # Pre-relax screening
    from idpforge.utils.pre_relax import (
        check_backbone_continuity, _JUNCTION_CA_THRESHOLD, _BACKBONE_CA_THRESHOLD,
    )

    written_files = []
    file_idx = counter

    # Indices that already have validated files
    existing_indices = set()
    if save_path is not None:
        from glob import glob as _glob
        for f in _glob(os.path.join(save_path, "*_validated.pdb")):
            base = os.path.basename(f).split("_")[0]
            if base.isdigit():
                existing_indices.add(int(base))

    for i in select_idx:
        if counter_cap is not None and file_idx > counter_cap:
            break

        # Skip indices that already have a relaxed file
        while file_idx in existing_indices:
            file_idx += 1
            if counter_cap is not None and file_idx > counter_cap:
                break
        if counter_cap is not None and file_idx > counter_cap:
            break

        aa = output["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = output["residue_index"][i] + 1

        # NaN coordinate filter
        if np.isnan(pred_pos).any():
            if verbose:
                print(f"       [NaN] Conformer {file_idx} has NaN coordinates, skipping.", flush=True)
            continue

        # Backbone continuity filter (pre-relaxation)
        if viol_mask is not None and len(viol_mask) == pred_pos.shape[0]:
            cont_pass, cont_details = check_backbone_continuity(
                pred_pos, viol_mask, residue_index=resid)
            if verbose and cont_details:
                # Monotonic attempt counter
                output_to_pdb._attempt = getattr(output_to_pdb, "_attempt", 0) + 1
                n = output_to_pdb._attempt
                verdict = "ACCEPTED" if cont_pass else "REJECTED"

                lines = []
                # Longest backbone break
                bbs = [d for d in cont_details if d["kind"] == "backbone"]
                if bbs:
                    bw = max(bbs, key=lambda d: d["distance"])
                    a, b = bw["res_i"], bw["res_j"]
                    region = "IDR" if viol_mask[a] else "fold"
                    op = "<=" if bw["pass"] else ">"
                    lines.append(f"longest backbone res {_disp_resid(a)}->{_disp_resid(b)} ({region}) "
                                 f"CA-CA = {bw['distance']:.2f} A {op} {_BACKBONE_CA_THRESHOLD} A "
                                 f"[{'PASS' if bw['pass'] else 'FAIL'}]")
                # Junction status
                juncs = [d for d in cont_details if d["kind"] == "junction"]
                if juncs:
                    jw = max(juncs, key=lambda d: d["distance"])
                    a, b = jw["res_i"], jw["res_j"]
                    ta = "IDR" if viol_mask[a] else "fold"
                    tb = "IDR" if viol_mask[b] else "fold"
                    op = "<=" if jw["pass"] else ">"
                    lines.append(f"junction res {_disp_resid(a)}({ta})<->{_disp_resid(b)}({tb}) "
                                 f"CA-CA = {jw['distance']:.2f} A {op} {_JUNCTION_CA_THRESHOLD} A "
                                 f"[{'PASS' if jw['pass'] else 'FAIL'}]")
                if not lines:
                    lines.append("(no backbone bonds checked)")
                lines[-1] += f" -- {verdict}"
                for k, L in enumerate(lines):
                    pre = "\n       [JUNCTION]" if k == 0 else "       [JUNCTION]"
                    print(f"{pre} Conformer {n}: {L}", flush=True)
            if not cont_pass:
                continue

        # Build full atom37 protein object
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=np.zeros(pred_pos.shape[:2]),
            chain_index=np.zeros(pred_pos.shape[0], dtype=int),
        )

        pdb_str = to_pdb(pred)

        # Numbered filename
        if save_path is not None:
            fname = os.path.join(save_path, f"{file_idx}_raw.pdb")
            with open(fname, "w") as f:
                f.write(pdb_str)
            written_files.append(fname)
        else:
            written_files.append(pdb_str)

        file_idx += 1

    # Relaxation with structural repair + validation
    if relax is not None:
        from openmm.app import PDBFile as OMMPDBFile
        from idpforge.utils.relax import relax_protein
        from idpforge.utils.structure_validation import (
            validate_structure_post_relax, check_bond_integrity,
            _check_fold_curvature, _check_junction_curvature,
        )
        from idpforge.utils.structure_repair import (
            repair_chirality, fix_histidine_naming,
        )

        _HIS_RESNAMES = {'HIS', 'HID', 'HIE', 'HIP'}
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        relaxed_files = []
        validate_attempt = 0
        valid_count = 0

        for raw_path in written_files:
            stem = os.path.basename(raw_path).replace("_raw.pdb", "")
            relaxed_path = os.path.join(save_path, stem + "_relaxed.pdb")
            validate_attempt += 1

            if verbose:
                print("-" * 60, flush=True)
                print(f"     [Batch Attempt {validate_attempt}] Conformer {stem}", flush=True)

            # --- Initial relaxation ---
            with open(raw_path) as _raw_fp:
                _raw_pdb_str = _raw_fp.read()
            unrelaxed_prot = from_pdb_string(_raw_pdb_str)
            del _raw_pdb_str
            relax_protein(
                relax, device_str,
                unrelaxed_prot,
                save_path, stem, **kwargs
            )
            del unrelaxed_prot
            if os.path.isfile(raw_path):
                os.remove(raw_path)

            if not os.path.isfile(relaxed_path):
                if verbose:
                    print(f"       [RELAX] Relaxation failed, skipping.", flush=True)
                continue

            # --- Pre-validation: Structural Repair ---
            needs_rerelax = False

            # 1a. Bond integrity -> find broken HIS residues
            chk = None
            try:
                chk = OMMPDBFile(relaxed_path)
                broken = check_bond_integrity(chk.topology, chk.positions)
                his_resids = {b['resid'] for b in broken
                              if b['resname'] in _HIS_RESNAMES
                              or b.get('resname2', '') in _HIS_RESNAMES}
            except Exception:
                his_resids = set()
            finally:
                if chk is not None:
                    del chk

            # 1b. Chirality repair
            n_chiral = repair_chirality(relaxed_path)
            if n_chiral > 0:
                if verbose:
                    print(f"       [REPAIR] Flipped {n_chiral} D-isomer(s)", flush=True)
                needs_rerelax = True

            # 1c. HIS naming fix
            if his_resids:
                n_his = fix_histidine_naming(relaxed_path, his_resids)
                if n_his > 0:
                    needs_rerelax = True

            # --- Re-relax if repairs applied ---
            if needs_rerelax:
                if verbose:
                    print(f"       [RE-RELAX] Re-relaxing after repairs...", flush=True)
                with open(relaxed_path) as _rel_fp:
                    _rel_pdb_str = _rel_fp.read()
                repaired_prot = from_pdb_string(_rel_pdb_str)
                del _rel_pdb_str
                os.remove(relaxed_path)
                relax_protein(
                    relax, device_str, repaired_prot,
                    save_path, stem, **kwargs
                )
                del repaired_prot
                if not os.path.isfile(relaxed_path):
                    if verbose:
                        print(f"       [RE-RELAX] Failed, discarding.", flush=True)
                    continue
            elif verbose:
                print(f"       [PRE-VALIDATION] All checks clean.", flush=True)

            # --- Structural validation ---
            t0 = time.perf_counter()
            chk = None
            try:
                chk = OMMPDBFile(relaxed_path)
                is_valid, info = validate_structure_post_relax(
                    chk.topology, chk.positions, pdb_path=relaxed_path,
                    full_report=True,
                    expected_knot_type=expected_knot_type
                )
            except Exception as e:
                is_valid = False
                info = {"reason": str(e), "chirality_pass": False,
                        "bonds_pass": False, "clash_pass": False,
                        "knot_pass": False}
            finally:
                if chk is not None:
                    del chk
            elapsed = time.perf_counter() - t0

            if verbose:
                chiral_pass = info.get("chirality_pass", True)
                bonds_pass = info.get("bonds_pass", True)
                clash_pass = info.get("clash_pass", True)
                knot_pass = info.get("knot_pass", True)
                clash_count = info.get("num_clashes", "?")
                clash_score = info.get("clash_score", 0.0)
                n_broken = info.get("num_broken_bonds", 0)
                knot_type = info.get("knot_type", "None")
                threshold = 10.0

                chiral_str = "PASS" if chiral_pass else "FAIL (D-Amino detected)"
                bonds_str = "PASS" if bonds_pass else \
                    f"FAIL ({info.get('broken_bonds_first_res', str(n_broken) + ' broken')})"
                clash_str = "PASS" if clash_pass else "FAIL"
                if expected_knot_type is not None:
                    detected = info.get("detected_knot_type")
                    disp = info.get("expected_knot_display") or str(expected_knot_type)
                    if knot_pass:
                        knot_str = f"PASS (native {disp})"
                    else:
                        fail_reason = info.get("knot_fail_reason", "")
                        knot_str = f"FAIL (expected {disp}, got {detected}) [{fail_reason}]"
                else:
                    knot_str = "PASS" if knot_pass else f"FAIL ({knot_type})"

                print(f"       [TIMING] Validation: {elapsed:.2f}s", flush=True)
                print(f"       [POST-MIN CHECK] Validation Results", flush=True)
                print(f"         - Chirality: {chiral_str}", flush=True)
                print(f"         - Bonds:     {bonds_str}", flush=True)
                print(f"         - Clashes:   {clash_count}  (Score: {clash_score:.2f} | Limit: {threshold:.1f})", flush=True)
                print(f"         - Topology:  {knot_str}", flush=True)

            # --- Post-relax structural gates (on the relaxed structure) ---
            curv_on = (fold_curv_ratio is not None and fold_curv_ratio > 0
                       and fold_ref_ca is not None and fold_mask is not None)
            kappa_on = (junction_kappa is not None and junction_kappa > 0
                        and viol_mask is not None)
            if is_valid and (curv_on or kappa_on):
                n_res = len(fold_mask) if fold_mask is not None else len(viol_mask)
                rel_ca = _ca_from_pdb_file(relaxed_path, n_res)

                # Folded-domain curvature gate
                if curv_on:
                    if rel_ca.shape[0] != np.asarray(fold_ref_ca).shape[0]:
                        is_valid = False
                        info["reason"] = info.get("reason", "") + " | fold-gate length mismatch"
                        if verbose:
                            print(f"         - Fold curvature: FAIL (residue count mismatch)", flush=True)
                    else:
                        gate_pass, gate_lines, gate_reason = _check_fold_curvature(
                            rel_ca, fold_ref_ca, fold_mask, fold_curv_ratio, window=fold_curv_window)
                        if not gate_pass:
                            is_valid = False
                            info["reason"] = info.get("reason", "") + " | " + gate_reason
                        if verbose:
                            detail = f"  ({gate_lines[0]})" if gate_lines else ""
                            print(f"         - Fold curvature: {'PASS' if gate_pass else 'FAIL'}{detail}", flush=True)

                # Junction curvature gate
                if kappa_on and len(viol_mask) == rel_ca.shape[0]:
                    curv_pass, curv_details = _check_junction_curvature(
                        rel_ca, viol_mask, junction_kappa)
                    if not curv_pass:
                        is_valid = False
                        info["reason"] = info.get("reason", "") + " | junction stretched (low curvature)"
                    if verbose and curv_details:
                        worst = min(curv_details, key=lambda d: d["mean_kappa"])
                        print(f"         - Junction curvature: {'PASS' if curv_pass else 'FAIL'}"
                              f"  (IDR res {_disp_resid(worst['idr_lo'])}-{_disp_resid(worst['idr_hi'])} mean kappa "
                              f"{worst['mean_kappa']:.3f} >= {junction_kappa} A^-1)", flush=True)

            if is_valid:
                valid_count += 1
                validated_path = relaxed_path.replace("_relaxed.pdb", "_validated.pdb")
                os.rename(relaxed_path, validated_path)
                # Stamp the final conformer as a clean MODEL block
                finalize_model_pdb(validated_path, stem)
                if verbose:
                    print(f"       [RESULT] SUCCESS!", flush=True)
                relaxed_files.append(validated_path)
            else:
                reason = info.get("reason", "Unknown")
                if verbose:
                    print(f"       [RESULT] FAILED ({reason}) [Thresh: 10.00]", flush=True)
                os.remove(relaxed_path)

            gc.collect()

        return relaxed_files

    return written_files


def finalize_model_pdb(path, model_idx):
    """Rewrite a single-conformer PDB in place as one clean MODEL block."""
    keep = ("ATOM", "HETATM", "TER", "ANISOU")
    with open(path) as fh:
        body = [ln.rstrip() + "\n" for ln in fh if ln.startswith(keep)]
    with open(path, "w") as fh:
        fh.write(f"MODEL     {int(model_idx):>4d}\n")
        fh.writelines(body)
        fh.write("ENDMDL\n")
