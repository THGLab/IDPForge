import numpy as np
from openfold.np.residue_constants import (
    rigid_group_atom_positions,
    van_der_waals_radius,
    restype_3to1,
)

# including masked type
sstypes = ["H", "E", "P", "A", "B", "C", "L", "-"]  
coil_types = ["A", "B", "C", "L"]
sstype_num = len(sstypes)
sstype_order = {sstype: i for i, sstype in enumerate(sstypes)}
ss_lib = {v: k for k, v in sstype_order.items()}
coil_sample_probs = {"A": 0.25, "B": 0.45, "L": 0.1, "C": 0.2} 

# ideal N, CA, C initial coordinates
init_N = np.array([-0.5272, 1.3593, 0.000])
init_CA = np.zeros_like(init_N)
init_C = np.array([1.525, 0.000, 0.000])

norm_N = init_N / (np.linalg.norm(init_N, axis=-1) + 1e-5)
norm_C = init_C / (np.linalg.norm(init_C, axis=-1) + 1e-5)
cos_ideal_NCAC = np.sum(norm_N * norm_C, axis=-1) # cosine of ideal N-CA-C bond angle

backbone_radius = np.array([van_der_waals_radius[a] for a in ["N", "C", "C"]])
# openfold rigids has O in local frame; okay as O not used in x_t featurization
backbone_atom_positions = {restype_3to1[k]: np.stack([a[-1] for a in v[:5]], axis=0) \
                            for k, v in rigid_group_atom_positions.items()}
backbone_atom_positions["G"] = np.array([[-0.572,  1.360,  0.000], [ 0.000,  0.000,  0.000], 
        [ 1.525,  0.000,  0.000], [-0.378, -0.560, -0.868], [ 2.155, -1.059, 0.000]])
# flip to N, CA, C, O, CB order
backbone_atom_positions = {k: v[[0, 1, 2, 4, 3]] for k, v in backbone_atom_positions.items()}

