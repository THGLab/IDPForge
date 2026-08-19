# Adapted from https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/script_utils.py
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited

import gc
import logging
import os
import time
import numpy as np
from openfold.np import residue_constants, protein
from openfold.np.relax import relax
from openmm import OpenMMException

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

ring_AA = {residue_constants.restypes.index(a) for a in ["F", "Y", "W", "H", "P"]}


def relax_protein(config, model_device, unrelaxed_protein,
        output_dir, pdb_name, viol_threshold=0.02, viol_mask=None):
    amber_relaxer = relax.AmberRelaxation(
        use_gpu=(model_device != "cpu"), **config,
    )

    t = time.perf_counter()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
    if type(model_device) is int:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(model_device)
    elif "," in model_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = model_device

    struct_str = None
    viol = None
    try:
        try:
            struct_str, _, viol = amber_relaxer.process(prot=unrelaxed_protein, cif_output=False)
        except (ValueError, OpenMMException) as e:
            logger.info("Minimization failed...abort")
            print(f"       [RELAX] REJECT: minimization failed | {e}", flush=True)
            return 0
        aatype = unrelaxed_protein.aatype
        # Accept gate over the free residues (IDR + junctions)
        if viol_mask is None:
            viol_mask = np.ones(len(viol))
        free = viol_mask.astype(bool)
        n_free = int(free.sum())
        if n_free == 0:
            free = np.ones(len(viol), dtype=bool)
            n_free = len(viol)
        viol_free = viol * free

        # Ring-residue violation gate (free set)
        n_ring = sum(1 for a, f in zip(aatype, free) if f and a in ring_AA)
        ring_res = [f"{residue_constants.restypes[a]}{int(r)}"
                    for a, v, r in zip(aatype, viol_free, unrelaxed_protein.residue_index)
                    if bool(v) and a in ring_AA]
        if len(ring_res) > viol_threshold * n_ring:
            shown = ", ".join(ring_res[:10]) + (f" +{len(ring_res) - 10} more" if len(ring_res) > 10 else "")
            print(f"       [RELAX] REJECT: ring-AA {len(ring_res)}/{n_ring} viol "
                  f"(allow {int(viol_threshold * n_ring)}) | {shown}", flush=True)
            return 0

        # Accept gate over the free set
        viol_total = int(sum(viol_free))
        viol_frac = viol_total / n_free
        viol_cap = max(4, int(np.ceil(viol_threshold * n_free)))
        if viol_frac > viol_threshold or viol_total > viol_cap:
            reasons = []
            if viol_frac > viol_threshold:
                reasons.append(f"frac {viol_frac:.1%}>{viol_threshold:.0%}")
            if viol_total > viol_cap:
                reasons.append(f"count {viol_total}>cap {viol_cap}")
            print(f"       [RELAX] REJECT: {' & '.join(reasons)} | {viol_total}/{n_free} free res viol", flush=True)
            return 0

        # Restore CUDA_VISIBLE_DEVICES only if it was set on entry
        if visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        relaxation_time = time.perf_counter() - t
        relaxed_output = os.path.join(output_dir, f'{pdb_name}_relaxed.pdb')
        with open(relaxed_output, 'w') as fp:
            fp.write(struct_str)
        print(f"       [RELAX] Saved {relaxed_output} ({relaxation_time:.2f}s)", flush=True)
        return 1
    finally:
        del amber_relaxer
        struct_str = None
        viol = None
        gc.collect()
