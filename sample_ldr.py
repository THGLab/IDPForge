# Sampling script for local disordered regions (LDR) + folded template.

import gc
import os
import sys
import yaml
import torch
import numpy as np
import pickle
import pandas as pd
import random
from importlib.metadata import version
import time
from glob import glob

import ml_collections as mlc
from pytorch_lightning import seed_everything

from idpforge.utils.diff_utils import Denoiser, Diffuser
from idpforge.model import IDPForge
from idpforge.misc import output_to_pdb
from idpforge.utils.prep_sec import fetch_sec_from_seq

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)

old_params = ["trunk.structure_module.ipa.linear_q_points.weight", "trunk.structure_module.ipa.linear_q_points.bias", "trunk.structure_module.ipa.linear_kv_points.weight", "trunk.structure_module.ipa.linear_kv_points.bias"]
seed_everything(42)

def combine_sec(fold_ss, idr_ss, mask):
    return "".join(fold_ss[i] if m else idr_ss[i] for i, m in enumerate(mask))

def main(ckpt_path, fold_template, output_dir, sample_cfg,
        batch_size=32, nsample=200, attn_chunk_size=None,
        device="cpu", ss_db_path=None, no_relax=False, verbose=False,
        fold_curv_ratio=0.0, fold_curv_window=15, junction_kappa=0.0,
        expected_knot_type=None):

    # 1. Load Config
    print(f"[ldr] Loading Config: {sample_cfg}", flush=True)
    settings = yaml.safe_load(open(sample_cfg, "r"))

    # 2. Setup Diffusion
    diffuser = Diffuser(settings["diffuse"]["n_tsteps"],
        euclid_b0=settings["diffuse"]["euclid_b0"], euclid_bT=settings["diffuse"]["euclid_bT"],
        tor_b0=settings["diffuse"]["torsion_b0"], tor_bT=settings["diffuse"]["torsion_bT"])
    denoiser = Denoiser(settings["diffuse"]["inference_steps"], diffuser)

    # 3. Initialize Model
    model = IDPForge(settings["diffuse"]["n_tsteps"],
        settings["diffuse"]["inference_steps"],
        mlc.ConfigDict(settings["model"]),
    )

    # Optional attention chunking
    if attn_chunk_size is not None:
        model.set_chunk_size(attn_chunk_size)

    # 4. Load Weights
    print(f"[ldr] Loading Weights...", flush=True)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if int(version("openfold").split(".")[0]) > 1:
        sd = {k.replace("points.", "points.linear.") if k in old_params else k: v for k, v in pl_sd["ema"]["params"].items()}
    else:
        sd = {k: v for k, v in pl_sd["ema"]["params"].items()}
    model.load_state_dict(sd)

    if device=="cuda": model.cuda()
    else: model.cpu()
    model.eval()

    # 5. Load Data
    fold_data = np.load(fold_template)
    sequence = str(fold_data["seq"])

    # 6. Prepare Secondary Structure
    if ss_db_path is not None and os.path.exists(ss_db_path):
        with open(ss_db_path, "rb") as f:
            pkl = pickle.load(f)
        if isinstance(pkl, (tuple, list)):
            SEC_database = pd.DataFrame({"sequence": pkl[1], "sec": pkl[0]})
        else:
            SEC_database = pd.DataFrame(pkl)
        try:
            s1 = fetch_sec_from_seq(sequence, nsample*2, SEC_database)
        except:
            s1 = ["C" * len(sequence)] * (nsample * 2)
    elif settings["sec_path"] is None:
        s1 = ["C" * len(sequence)] * (nsample * 2)
    else:
        with open(settings["sec_path"], "r") as f:
            ss_lines = f.read().split("\n")
        seq_len = len(sequence)
        s1 = [s[:seq_len] for s in ss_lines if len(s) >= seq_len]

    ss = [combine_sec(str(fold_data["sec"]), d, fold_data["mask"]) for d in s1 if len(d)>sum(~fold_data["mask"])]
    crd_offset = fold_data.get("coord_offset", None)

    # 7. Relaxation Config
    if no_relax:
        relax_opts = None
        search_pattern = "*_raw.pdb"
    else:
        relax_config = settings["relax"]
        # Exclude the IDR from the positional restraint (fold stays restrained)
        relax_config["exclude_residues"] = np.where(~fold_data["mask"])[0].tolist()
        relax_opts = mlc.ConfigDict(relax_config)
        search_pattern = "*_validated.pdb"

    # Output Setup
    os.makedirs(output_dir, exist_ok=True)
    abs_output_dir = os.path.abspath(output_dir)

    # Emit the graft spec for a size-capped template
    _gfiles = getattr(fold_data, "files", [])
    if "graft_offset" in _gfiles and ("graft_fold_range" in _gfiles or "graft_fold_ranges" in _gfiles):
        import json as _json
        try:
            _spec = {
                "offset": int(fold_data["graft_offset"]),
                "idr_ranges": np.asarray(fold_data["graft_idr_ranges"]).tolist(),
                "note": "Auto-emitted by sample_ldr from a size-capped template; Step 4 grafts "
                        "the dropped fold back on, then relaxes + validates.",
            }
            if "graft_static_pdb" in _gfiles:
                _spec["static_pdb"] = str(fold_data["graft_static_pdb"])
            if "graft_fold_ranges" in _gfiles:
                _spec["fold_ranges"] = np.asarray(fold_data["graft_fold_ranges"]).tolist()
            else:
                _spec["fold_range"] = np.asarray(fold_data["graft_fold_range"]).tolist()
            with open(os.path.join(abs_output_dir, "_truncation.json"), "w") as _sf:
                _json.dump(_spec, _sf, indent=2)
        except Exception as _e:
            print(f"[ldr] WARNING: could not emit _truncation.json graft spec: {_e}", flush=True)

    # Step-2 continuation pool: emit the origin
    if "step2_a_conformer" in _gfiles:
        import json as _json2
        try:
            _spec2 = {"a_conformer": str(fold_data["step2_a_conformer"]),
                      "a_len": int(fold_data["step2_a_len"]),
                      "b_len": int(fold_data["step2_b_len"])}
            _spec2["direction"] = (str(fold_data["step2_direction"])
                                   if "step2_direction" in _gfiles else "append")
            with open(os.path.join(abs_output_dir, "_step2.json"), "w") as _sf:
                _json2.dump(_spec2, _sf, indent=2)
        except Exception as _e:
            print(f"[ldr] WARNING: could not emit _step2.json: {_e}", flush=True)

    def count_done():
        return len(glob(os.path.join(abs_output_dir, search_pattern)))

    def next_available_idx():
        """Find the smallest positive integer not already used."""
        existing_files = glob(os.path.join(abs_output_dir, search_pattern))
        used = set()
        for f in existing_files:
            base = os.path.basename(f).split("_")[0]
            if base.isdigit():
                used.add(int(base))
        idx = 1
        while idx in used:
            idx += 1
        return idx

    current_count = count_done()
    print(f"[ldr] Found {current_count} existing files. Target: {nsample}", flush=True)

    total_diffused = 0
    while current_count < nsample:
        chunk = min(batch_size, nsample - current_count)
        if chunk < 1: chunk = 1
        total_diffused += chunk

        seq_list = [sequence] * chunk
        ss_list = random.sample(ss, chunk)

        # Init Noise
        xt_list, tor_list = denoiser.init_samples(seq_list)

        # Init Template
        template = {k: torch.tensor(np.tile(v[None, ...], (chunk,) + (1,) * len(v.shape)),
            device=model.device, dtype=torch.long if k=="mask" else torch.float)
            for k, v in fold_data.items() if k in ["torsion", "mask"]}

        if crd_offset is None:
            template["coord"] = torch.tensor(fold_data["coord"], device=model.device, dtype=torch.float)
        else:
            template["coord"] = torch.tensor(fold_data["coord"][None, ...] - crd_offset[np.random.choice(crd_offset.shape[0],
                chunk, replace=False)][:, None, None, :], device=model.device, dtype=torch.float)

        # Inference
        start_idx = next_available_idx()
        print(f"[ldr] Generating batch of {chunk} starting at idx {start_idx} "
              f"(progress: {current_count}/{nsample})...", flush=True)
        # bf16 autocast for the GPU forward (CUDA only)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                              enabled=(device == "cuda")):
            outputs = model.sample(denoiser, seq_list, ss_list, tor_list, xt_list,
                template_cfgs=template)

        # CA reference for the fold-curvature gate
        fold_ref_ca = fold_data["coord"][..., 1, :]
        if fold_ref_ca.ndim == 3:
            fold_ref_ca = fold_ref_ca[0]

        output_to_pdb(outputs, relax=relax_opts,
                save_path=abs_output_dir, counter=start_idx,
                counter_cap=nsample, viol_mask=~fold_data["mask"],
                verbose=verbose,
                fold_ref_ca=fold_ref_ca,
                fold_mask=fold_data["mask"],
                fold_curv_ratio=fold_curv_ratio,
                fold_curv_window=fold_curv_window,
                junction_kappa=junction_kappa,
                expected_knot_type=expected_knot_type,
                orig_resid=(fold_data["trunc_orig_idx"]
                            if "trunc_orig_idx" in fold_data else None))

        del outputs, template, seq_list, ss_list, xt_list, tor_list
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Re-count files on disk
        current_count = count_done()

    print(f"[ldr] Generation Complete. {current_count} validated conformers "
          f"from {total_diffused} diffused in {abs_output_dir}")
    # Stats line for the Step-3 orchestrator
    print(f"[LDR_STATS] diffused={total_diffused} validated={current_count}", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path')
    parser.add_argument('fold_input')
    parser.add_argument('out_dir')
    parser.add_argument('sample_cfg')
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--nconf', default=100, type=int)
    parser.add_argument('--attention_chunk', default=None, type=int,
                        help="Attention chunk size passed to model.set_chunk_size "
                             "(lower = less memory, slower). Omit to disable chunking.")
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--ss_db', default=None, type=str)
    parser.add_argument('--no_relax', action="store_true", help="Skip relaxation (outputs raw pdb)")
    parser.add_argument('--fold_curv_ratio', default=0.0, type=float,
        help="Reject conformers whose junction-adjacent folded domain straightens below this fraction "
             "of the template's curvature (rigid-motion invariant; replaces folded-lDDT; 0 disables).")
    parser.add_argument('--fold_curv_window', default=15, type=int,
        help="Folded residues into the fold from each junction to average for the curvature gate.")
    parser.add_argument('--junction_kappa', default=0.0, type=float,
        help="Reject conformers whose junction-adjacent IDR (last ~20 residues) is pulled taut to "
             "reach the anchor -- mean backbone curvature kappa below this (A^-1; 0 disables).")
    parser.add_argument('--expected_knot_type', default=None, type=str,
                        help="Per-domain knot spec as a JSON string "
                             '(e.g. \'[{"range":[65,280],"knot":null},'
                             '{"range":[320,450],"knot":"3_1"}]\'). '
                             "An empty list means no topological constraint. "
                             "Omit the flag to apply the legacy full-chain unknot baseline.")
    parser.add_argument('--verbose', action="store_true", help="Print structural validation details")

    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # Parse the per-domain knot spec JSON at the CLI boundary
    expected_spec = None
    if args.expected_knot_type is not None:
        import json as _json
        try:
            expected_spec = _json.loads(args.expected_knot_type)
        except _json.JSONDecodeError as exc:
            sys.exit(f"[ldr] ERROR: --expected_knot_type must be valid JSON "
                     f"(got {args.expected_knot_type!r}): {exc}")
        if not isinstance(expected_spec, list):
            sys.exit(f"[ldr] ERROR: --expected_knot_type JSON must be a list, "
                     f"got {type(expected_spec).__name__}")

    main(args.ckpt_path, args.fold_input, args.out_dir, args.sample_cfg,
         args.batch, args.nconf,
         attn_chunk_size=args.attention_chunk,
         device=device,
         ss_db_path=args.ss_db,
         no_relax=args.no_relax,
         verbose=args.verbose,
         fold_curv_ratio=args.fold_curv_ratio,
         fold_curv_window=args.fold_curv_window,
         junction_kappa=args.junction_kappa,
         expected_knot_type=expected_spec)
