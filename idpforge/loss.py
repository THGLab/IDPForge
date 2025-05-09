import torch
from torch import nn

from openfold.np.residue_constants import restype_order
from openfold.utils.rigid_utils import Rigid, Rotation, identity_rot_mats
from openfold.utils.loss import (
    backbone_loss,
    compute_fape,
    distogram_loss,
    find_structural_violations,
    violation_loss
)
from openfold.data.data_transforms import pseudo_beta_fn
from idpforge.utils.tensor_utils import get_dih, get_chi_mask, align_rigids


def alt_beta_fn(aatype, all_atom_positions):
    """Custom CB index based on input data format"""
    is_gly = torch.eq(aatype, restype_order["G"])
    ca_idx = 1
    cb_idx = 4
    pseudo_beta = torch.where(
        torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )
    return pseudo_beta

def sidechain_fape(pred_rigids, true_rigids, pred_coords, true_coords, 
                   atom_mask, sidechain_mask, fape_clamp=None):
    return compute_fape(
            Rigid.from_tensor_4x4(pred_rigids)[..., 2:],
            Rigid.from_tensor_4x4(true_rigids[..., 1:, :, :]),
            sidechain_mask,
            pred_coords[..., :9, :],
            true_coords, atom_mask[..., :9],
            l1_clamp_distance=fape_clamp,
            length_scale=10)
  

def calc_loss(pred, true_rigids, true_coords, true_torsions,
              loss_cfg):
    torsion_mask = torch.cat((pred["atom14_atom_exists"][..., :1].repeat(1, 1, 3),
        get_chi_mask(pred["aatype"]) * pred["atom14_atom_exists"][..., :1]), dim=-1)
    angloss = torsion_loss(pred["positions"][-1][..., :3, :],
                           pred["angles"][-1],
                           pred["unnormalized_angles"][-1],
                           true_torsions,
                           mask=torsion_mask)
    torsion_mask = torsion_mask[..., 1:]
    loss_dict = {"angular": angloss}

    if loss_cfg["weights"]["fape"] > 0:
        fapeloss = backbone_loss(traj=pred["frames"],
            backbone_rigid_tensor=true_rigids[..., 0, :, :],
            backbone_rigid_mask=pred["atom14_atom_exists"][..., 0],
            use_clamped_fape=loss_cfg["fape"]["use_clamp"],
            clamp_distance=loss_cfg["fape"]["clamp_distance"])   # CA only

        sidechain_fapeloss = sidechain_fape(
                pred["sidechain_frames"][-1], true_rigids, 
                pred_coords=pred["positions"][-1], 
                true_coords=true_coords,
                atom_mask=pred["atom14_atom_exists"], 
                sidechain_mask=torsion_mask, 
                fape_clamp=None).mean()
        loss_dict["fape"] = (fapeloss + sidechain_fapeloss) ** 2
    
    if loss_cfg["weights"]["dist"] > 0:
        dloss = cb_dist_loss(pred, true_coords,
                    loss_cfg["dist"]["loop_clamp"])
        if loss_cfg["dist"]["sidechain"] > 0:
            dloss = dloss + loss_cfg["dist"]["sidechain"] * dist_loss(
                    pred["positions"][-1][..., :9, :], true_coords,
                    pred["atom14_atom_exists"][..., :9],
                    loss_cfg["dist"]["sidechain_clamp"])
        loss_dict["dist"] = dloss
                
    if loss_cfg["weights"]["violation"] > 0:
        violoss = viol_loss(pred)
        loss_dict["violation"] = violoss

    return torch.sum(torch.stack([loss_cfg["weights"][k]*v for k, v in loss_dict.items()])), \
        {name:loss_dict[name].item() for name in loss_dict}


# based on RFdiffusion
def mse_frame_loss(
        pred_frames: Rigid,
        target_frames: Rigid,
        frames_mask: torch.Tensor,
        l1_clamp_distance = None,
        length_scale = 10,
        eps = 1e-8, w_rot = 0.5,
    ) -> torch.Tensor:
    # align global frames
    target_frames = align_rigids(pred_frames, target_frames, frames_mask) 
    # [*, *, N_res]
    trans_error = torch.sqrt(
            torch.sum((pred_frames.get_trans() - target_frames.get_trans())**2, dim=-1) + eps
    )
    if l1_clamp_distance is not None:
        trans_error = torch.clamp(trans_error, min=eps, max=l1_clamp_distance)
    # [*, *, N_res]
    trans_error = trans_error ** 2 / length_scale
    
    # [*, *, N_res, 3, 3]
    rot_mul = pred_frames.get_rots().invert().compose_r(target_frames.get_rots()).get_rot_mats()
    identity_rot = identity_rot_mats(rot_mul.shape[:-2], dtype=rot_mul.dtype, 
            device=rot_mul.device, requires_grad=False)
    
    # [*, *, N_res]
    rot_error = torch.sum((identity_rot - rot_mul) ** 2, dim=(-1, -2))
    error = (trans_error + w_rot * rot_error) * frames_mask 
    error = torch.sum(error, dim=(-2, -1)) / torch.sum(frames_mask, dim=(-2, -1))
    return error


def cb_dist_loss(pred, true_coords, loop_clamp, eps=1e-6, l_coef=1.):
    beta = pseudo_beta_fn(pred["aatype"], pred["positions"][-1], None)
    true_beta = alt_beta_fn(pred["aatype"], true_coords)
    pred_d = torch.sqrt(torch.sum(
            (beta[..., None, :] - beta[..., None, :, :]) ** 2,
        dim=-1) + eps)
    true_d = torch.sqrt(torch.sum(
            (true_beta[..., None, :] - true_beta[..., None, :, :]) ** 2,
        dim=-1) + eps)
    
    square_mask = pred["atom14_atom_exists"][..., 0][..., None] * pred["atom14_atom_exists"][..., 0][..., None, :]
    ca_loss = ca_connectivity_loss(pred["positions"][-1][..., 1, :], 
            pred["atom14_atom_exists"][..., 1]) # reinforce peptide connection
    error = torch.clamp(true_d - pred_d, min=eps, max=loop_clamp)
    sec_mask = (pred["sstype"] < 2).float()
    sec_mask = sec_mask[..., None] * sec_mask[..., None, :] * square_mask
    sec_error = torch.clamp(pred_d - true_d, min=eps, max=loop_clamp)
    return ((square_mask * error**2).sum() / square_mask.sum() + ca_loss) * l_coef + \
            (sec_mask * sec_error**2).sum() / sec_mask.sum() 

def dist_loss(pred_coords, true_coords, pos_mask, clamp, eps=1e-6):
    pred_d = torch.sum(
            (pred_coords[..., None, :] - pred_coords[..., None, :, :]) ** 2,
        dim=-1) + eps
    true_d = torch.sum(
            (true_coords[..., None, :] - true_coords[..., None, :, :]) ** 2,
        dim=-1) + eps
    square_mask = pos_mask[..., None] * pos_mask[..., None, :]
    error = torch.clamp(torch.sqrt(true_d) - torch.sqrt(pred_d), min=-clamp, max=clamp)
    return (square_mask * error**2).sum() / square_mask.sum()


def torsion_loss(pred_coords, pred_norm_torsions, pred_torsions, true_torsions, mask):
    # [*, N, 7]
    l_angle_norm = torch.abs(torch.norm(pred_torsions, dim=-1) - 1).mean()
    l_torsion = torch.norm(pred_norm_torsions - true_torsions, dim=-1)
 
    return (l_torsion * mask).sum() / mask.sum() + 0.02 * l_angle_norm
        
def ca_connectivity_loss(coords, ca_mask):
    # reinforce peptide connection
    ca_dist = torch.sqrt(
        1e-6 + torch.sum(
            (coords[..., :-1, :] - coords[..., 1:, :])** 2,
        dim=-1)
    )
    ca_dist = ca_dist * ca_mask[..., 1:]
    return ((ca_dist - ca_ca)**2).sum() / ca_mask.sum()

def viol_loss(batch, return_metric=False):
    violations = find_structural_violations(batch, batch["positions"][-1].detach(), 5, 5)
    bond_clash_loss = violation_loss(violations, batch["atom14_atom_exists"]).mean()
    if return_metric:
        return bond_clash_loss, violations["total_per_residue_violations_mask"]
    return bond_clash_loss 

