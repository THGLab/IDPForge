# Adapted from https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/script_utils.py
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited

import logging
import os
import time

from openfold.np import residue_constants, protein
from openfold.np.relax import relax

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

ring_AA = {residue_constants.restypes.index(a) for a in ["F", "Y", "W", "H", "P"]}


def relax_protein(config, model_device, unrelaxed_protein, 
        output_dir, pdb_name, viol_threshold=0.025):
    amber_relaxer = relax.AmberRelaxation(
        use_gpu=(model_device != "cpu"), **config,
    )

    t = time.perf_counter()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
    if type(model_device) is int:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(model_device)
    elif "," in model_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = model_device 
    struct_str, _, viol = amber_relaxer.process(prot=unrelaxed_protein, cif_output=False)
    aatype = unrelaxed_protein.aatype
    if len(ring_AA.intersection({a for a, v in zip(aatype, viol) if bool(v)})) > 0:
        return 0
    if sum(viol)/len(viol) > viol_threshold or sum(viol) > 2:
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    relaxation_time = time.perf_counter() - t
    relaxed_output = os.path.join(output_dir, f'{pdb_name}_relaxed.pdb')
    with open(relaxed_output, 'w') as fp:
        fp.write(struct_str)
    logger.info(f"saved at {relaxed_output} with relaxation time: {relaxation_time}")
    return 1


