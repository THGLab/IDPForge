import os
import sys
import yaml
import torch
import random
import pickle
import pandas as pd
from glob import glob
from importlib.metadata import version

import ml_collections as mlc
from pytorch_lightning import seed_everything

from idpforge.model import IDPForge
from idpforge.utils.diff_utils import Denoiser, Diffuser
from idpforge.misc import output_to_pdb
from idpforge.utils.prep_sec import fetch_sec_from_seq

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)

old_params = ["trunk.structure_module.ipa.linear_q_points.weight", "trunk.structure_module.ipa.linear_q_points.bias", "trunk.structure_module.ipa.linear_kv_points.weight", "trunk.structure_module.ipa.linear_kv_points.bias"]
seed_everything(42)

def main(sequence, ckpt_path, output_dir, sample_cfg,
        batch_size=32, nsample=200, device="cpu", no_relax=False, verbose=False):

    # 1. Load Config
    print(f"[idp] Loading Config: {sample_cfg}", flush=True)
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

    # 4. Load Weights
    print(f"[idp] Loading Weights: {ckpt_path}", flush=True)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if int(version("openfold").split(".")[0]) > 1:
        sd = {k.replace("points.", "points.linear.") if k in old_params else k: v for k, v in pl_sd["ema"]["params"].items()}
    else:
        sd = {k: v for k, v in pl_sd["ema"]["params"].items()}
    model.load_state_dict(sd)

    if device=="cuda":
        model.cuda()
    else:
        model.cpu()
    model.eval()
    seq_len = len(sequence)

    # 5. Potential Config
    if settings["potential"]:
        potential_cfg = {"potential_type": [], "weights": {},  "potential_cfg": {},
            "timescale": settings["potential_cfg"].pop("timescale"),
            "grad_clip": settings["potential_cfg"].pop("grad_clip"),
        }
        for k in settings["potential_cfg"]:
            if k in ["pre", "noe"]:
                from idpforge.utils.np_utils import get_contact_map
                exp_pre = get_contact_map(settings["potential_cfg"]["pre"]["exp_path"],
                    seq_len)
                potential_cfg["potential_cfg"]["contact"] = {"contact_bounds": exp_pre,
                    "exp_mask_p": settings["potential_cfg"]["pre"]["exp_mask_p"]}
                potential_cfg["weights"]["contact"] = settings["potential_cfg"]["pre"].get("weight", 1)
                potential_cfg["potential_type"].append("contact")
            elif k == "rg":
                potential_cfg["potential_cfg"]["rg"] = {"target": settings["potential_cfg"]["rg"]["ens_avg"]}
                potential_cfg["weights"]["rg"] = settings["potential_cfg"]["rg"].get("weight", 1)
                potential_cfg["potential_type"].append("rg")

            else:
                raise NotImplementedError()
    else:
        potential_cfg = None

    # 6. Prepare Secondary Structure
    print(f"[idp] Preparing secondary structure...", flush=True)
    if settings["sec_path"] is None:
        with open(settings["data_path"], "rb") as f:
            pkl = pickle.load(f)
        SEC_database = pd.DataFrame({"sequence": pkl[1], "sec": pkl[0]})
        try:
            ss = fetch_sec_from_seq(sequence, nsample*2, SEC_database)
        except Exception as e:
            print(f"[idp] WARNING: fetch_sec_from_seq failed ({e}), falling back to all-coil", flush=True)
            ss = ["C" * seq_len] * (nsample * 2)
        del SEC_database
    else:
        with open(settings["sec_path"], "r") as f:
            ss = f.read().split("\n")
        ss = [s[:seq_len] for s in ss if len(s) >= seq_len]

    # 7. Relaxation Config
    if no_relax:
        relax_opts = None
        search_pattern = "*_raw.pdb"
    else:
        relax_opts = mlc.ConfigDict(settings["relax"])
        search_pattern = "*_validated.pdb"

    # 8. Output Setup
    os.makedirs(output_dir, exist_ok=True)
    abs_output_dir = os.path.abspath(output_dir)

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
    print(f"[idp] Found {current_count} existing files. Target: {nsample}", flush=True)

    # 9. Generation Loop
    while current_count < nsample:
        chunk = min(batch_size, nsample - current_count)
        if chunk < 1:
            chunk = 1

        seq_list = [sequence] * chunk
        ss_list = random.sample(ss, chunk)
        xt_list, tor_list = denoiser.init_samples(seq_list)

        start_idx = next_available_idx()
        print(f"[idp] Generating batch of {chunk} starting at idx {start_idx} "
              f"(progress: {current_count}/{nsample})...", flush=True)
        with torch.no_grad():
            outputs = model.sample(denoiser, seq_list, ss_list, tor_list, xt_list,
                    potential_cfgs=potential_cfg)

        output_to_pdb(outputs, relax=relax_opts,
                save_path=abs_output_dir, counter=start_idx, counter_cap=nsample,
                verbose=verbose)

        # Re-count actual files on disk (some conformers may be rejected by relaxation)
        current_count = count_done()

    print(f"[idp] Generation Complete. {current_count} validated conformers in {abs_output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('seq')
    parser.add_argument('ckpt_path')
    parser.add_argument('output_dir')
    parser.add_argument('sample_cfg')
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--nconf', default=100, type=int)
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--no_relax', action="store_true", help="Skip relaxation (outputs raw pdb)")
    parser.add_argument('--verbose', action="store_true", help="Print structural validation details")

    args = parser.parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    main(args.seq, args.ckpt_path, args.output_dir, args.sample_cfg,
         args.batch, args.nconf, device, no_relax=args.no_relax, verbose=args.verbose)
