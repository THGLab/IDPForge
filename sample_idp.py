import os
import yaml
import torch
import random

from glob import glob
import ml_collections as mlc
from pytorch_lightning import Trainer, seed_everything

from idpforge.model import IDPForge
from idpforge.utils.diff_utils import Denoiser, Diffuser
from idpforge.misc import output_to_pdb

seed_everything(42)

def main(ckpt_path, output_dir, sample_cfg,
        batch_size=32, nsample=200, device="cpu"):

    settings = yaml.safe_load(open(sample_cfg, "r"))
    diffuser = Diffuser(settings["diffuse"]["n_tsteps"],
        euclid_b0=settings["diffuse"]["euclid_b0"], euclid_bT=settings["diffuse"]["euclid_bT"],
        tor_b0=settings["diffuse"]["torsion_b0"], tor_bT=settings["diffuse"]["torsion_bT"])
    denoiser = Denoiser(settings["diffuse"]["inference_steps"], diffuser)
    model = IDPForge(settings["diffuse"]["n_tsteps"], 
        settings["diffuse"]["inference_steps"],
        mlc.ConfigDict(settings["model"]),
    ) 
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict({k: v for k, v in sd["ema"]["params"].items()})
    if device=="cuda":
        model.cuda()
    else:
        model.cpu()
    model.eval()
    seq_len = len(settings["sequence"])

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
    
    with open(settings["sec_path"], "r") as f:
        ss = f.read().split("\n")
    ss = [s[:seq_len] for s in ss if len(s) >= seq_len]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    start = len(glob(output_dir+"/*_relaxed.pdb"))
    while start < nsample:
        chunk = min(batch_size, nsample - start) 
        seq_list = [settings["sequence"]] * chunk
        ss_list = random.sample(ss, chunk)
        xt_list, tor_list = denoiser.init_samples(seq_list)
        outputs = model.sample(denoiser, seq_list, ss_list, tor_list, xt_list, 
                potential_cfgs=potential_cfg)
        
        output_to_pdb(outputs, relax=mlc.ConfigDict(settings["relax"]), 
                save_path=output_dir, counter=start+1, counter_cap=nsample)
        start = len(glob(output_dir+"/*_relaxed.pdb")) 

        
    print("done")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path')
    parser.add_argument('output_dir')
    parser.add_argument('sample_cfg')
    parser.add_argument('--batch', default=32, type=int) 
    parser.add_argument('--nconf', default=100, type=int)
    parser.add_argument('--cuda', action="store_true")

    args = parser.parse_args()
    main(args.ckpt_path, args.output_dir, args.sample_cfg,
         args.batch, args.nconf, "cuda" if args.cuda else "cpu")
