# Adapted from https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/script_utils.py
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited

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

    try:
        struct_str, _, viol = amber_relaxer.process(prot=unrelaxed_protein, cif_output=False)
    except (ValueError, OpenMMException):
        # ValueError: CYS protonation error; OpenMMException: NaN coords, etc.
        logger.info("Minimization failed...abort")
        return 0
    aatype = unrelaxed_protein.aatype
    violated_ring = ring_AA.intersection({a for a, v in zip(aatype, viol) if bool(v)})
    if len(violated_ring) > 0:
        names = [residue_constants.restypes[a] for a in violated_ring]
        print(f"       [RELAX] Rejected: ring AA violation ({', '.join(names)})", flush=True)
        return 0
    if viol_mask is None:
        viol_mask = np.ones(len(viol))
    rate = sum(viol*viol_mask.astype(float)) / max(sum(viol_mask), 1)
    total_viol = int(sum(viol))
    if rate > viol_threshold or total_viol > 4:
        print(f"       [RELAX] Rejected: IDR viol rate={rate:.3f} (limit {viol_threshold}) | "
              f"total={total_viol} (limit 4)", flush=True)
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    relaxation_time = time.perf_counter() - t
    relaxed_output = os.path.join(output_dir, f'{pdb_name}_relaxed.pdb')
    with open(relaxed_output, 'w') as fp:
        fp.write(struct_str)
    print(f"       [RELAX] Saved {relaxed_output} ({relaxation_time:.2f}s) | "
          f"IDR viol rate={rate:.3f} | total={total_viol}", flush=True)
    return 1


