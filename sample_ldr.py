# Sampling script for local disordered regions
# created by OZ, 1/8/25

import os
import yaml
import torch
from glob import glob
import numpy as np
import random
import ml_collections as mlc
from pytorch_lightning import Trainer, seed_everything

from idpforge.model import IDPForge
from idpforge.utils.diff_utils import Denoiser, Diffuser
from idpforge.misc import output_to_pdb

seed_everything(42)

def combine_sec(fold_ss, idr_ss, mask):
    idr_counter = 0
    ss = ""
    for fs, m in zip(fold_ss, mask):
        if m:
            ss += fs
        else:
            ss += idr_ss[idr_counter]
            idr_counter += 1
    return ss

def main(ckpt_path, fold_template, output_dir, sample_cfg,
        batch_size=32, nsample=200, attn_chunk_size=None, device="cpu"):

    settings = yaml.safe_load(open(sample_cfg, "r"))
    diffuser = Diffuser(settings["diffuse"]["n_tsteps"])
    denoiser = Denoiser(settings["diffuse"]["inference_steps"], diffuser)
    model = IDPForge(settings["diffuse"]["n_tsteps"], 
        settings["diffuse"]["inference_steps"], 
        mlc.ConfigDict(settings["model"]), t_end=settings["diffuse"]["end"],
    )
    if attn_chunk_size is not None:
        model.set_chunk_size(attn_chunk_size)
    else:
        attn_chunk_size = 0
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict({k: v for k, v in sd["ema"]["params"].items()})
    if device=="cuda":
        model.cuda()
    else:
        model.cpu()
    model.eval()
    
    fold_data = np.load(fold_template)
    sequence = str(fold_data["seq"])
    with open(settings["sec_path"], "r") as f:
        s1 = f.read().split("\n")
    ss = str(fold_data["sec"])
    ss = [combine_sec(ss, d, fold_data["mask"]) for d in s1 if len(d)>sum(~fold_data["mask"])]
    crd_offset = fold_data.get("coord_offset", None)

    relax_config = settings["relax"] 
    # use exclude_residues to apply restraints on folded structures
    relax_config["exclude_residues"] = np.where(fold_data["mask"])[0].tolist()
    viol = 0.025 if attn_chunk_size < len(sequence) else 0.025*sum(fold_data["mask"])/len(sequence)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    start = len(glob(output_dir+"/*_relaxed.pdb"))
    while start < nsample: 
        chunk = min(batch_size, nsample - start)
        seq_list = [sequence] * chunk 
        ss_list = random.sample(ss, chunk)
        xt_list, tor_list = denoiser.init_samples(seq_list, crd_offset)
        template = {k: torch.tensor(np.tile(v[None, ...], (chunk,) + (1,) * len(v.shape)), 
            device=model.device, dtype=torch.long if k=="mask" else torch.float) 
            for k, v in fold_data.items() if k in ["coord", "torsion", "mask"]}

        outputs = model.sample(denoiser, seq_list, ss_list, tor_list, xt_list, 
                template_cfgs=template)
        output_to_pdb(outputs, relax=mlc.ConfigDict(relax_config), 
                save_path=output_dir, counter=start+1,
                counter_cap=nsample, drop_viol=viol)
        start = len(glob(output_dir+"/*_relaxed.pdb"))
        
    print("done")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path')
    parser.add_argument('fold_input', help="prepared folded structural data in .npz from running init_ldr_template.py")
    parser.add_argument('out_dir')
    parser.add_argument('sample_cfg')
    parser.add_argument('--batch', default=32, type=int) 
    parser.add_argument('--nconf', default=100, type=int)
    parser.add_argument('--attention_chunk', default=None, type=int)
    parser.add_argument('--cuda', action="store_true")

    args = parser.parse_args()
    main(args.ckpt_path, args.fold_input, args.out_dir, args.sample_cfg,
         args.batch, args.nconf, 
         attn_chunk_size=args.attention_chunk,
         device="cuda" if args.cuda else "cpu")
