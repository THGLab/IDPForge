import torch
from openfold.np.residue_constants import chi_angles_mask
from openfold.utils.rigid_utils import Rigid, Rotation

def get_chi_mask(sequence):
    m = torch.tensor(chi_angles_mask, device=sequence.device, requires_grad=False)
    return m[sequence, ...]

    
def calc_norm(s, eps=1e-8, max_clamp=None, keepdim=False):
    return torch.sqrt(torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=keepdim), 
                            min=eps, max=max_clamp))

# credit https://github.com/RosettaCommons/RFdiffusion.git
# More complicated version splits error in CA-N and CA-C (giving more accurate CB position)
# It returns the rigid transformation from local frame to global frame
def rigid_from_3_points(xyz, eps=1e-8):
    # N, Ca, C - [B, L, 3]
    # R - [B, L, 3, 3], det(R)=1, inv(R) = R.T, R is a rotation matrix
    N = xyz[..., 0, :]
    Ca = xyz[..., 1, :]  # [1, L, 3, 3]
    C = xyz[..., 2, :]
    B, L = N.shape[:2]

    v1 = C - Ca
    v2 = N - Ca     
   
    e1 = v1 / calc_norm(v1, eps)
    u2 = v2 - (torch.einsum("bli, bli -> bl", e1, v2)[..., None] * e1)
    e2 = u2 / calc_norm(u2, eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)  # [B,L,3,3] - rotation matrix

    return R

def cross(a, b):
    """
    cross product at the last dimension to support bfloat16
    a , b [..., 3]
    """
    C = [a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
         a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
         a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]]
    
    return torch.stack(C, dim=-1)
    
def cdist(a, b):
    """
    calculates the euclidean distance between 2 matrices at the last dimension
    """
    # Calculate pairwise squared differences
    diff = a.unsqueeze(-3) - b.unsqueeze(-2)
    return torch.sqrt((diff**2).sum(dim=-1))
    

def generate_Cbeta(N, Ca, C):
    # recreate Cb given N, Ca, C
    b = Ca - N
    c = C - Ca
    a = cross(b, c)
    # from RosettaCommons
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    return Cb

def get_ang(a, b, c):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
    from Cartesian coordinates of three sets of atoms a,b,c 

    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """
    v = a - b
    w = c - b
    v /= torch.norm(v, dim=-1, keepdim=True)
    w /= torch.norm(w, dim=-1, keepdim=True)
    vw = torch.sum(v * w, dim=-1)

    return torch.acos(vw)
    
def get_dih(a, b, c, d, return_vec=False):
    """calculate dihedral angles for all consecutive quadruples (a[i], b[i], c[i], d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors or numpy array of shape [batch, nres, 3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor or numpy array of shape [batch, nres]
          stores resulting dihedrals
          if return_vec, [batch, nres, 2]
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c
    
    b1 = b1 / calc_norm(b1, keepdim=True)
 
    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1

    x = torch.sum(v * w, dim=-1)
    y = torch.sum(cross(b1, v) * w, dim=-1)
    if return_vec:
        return torch.stack((y, x), dim=-1)
    return torch.atan2(y, x)

    
def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 1000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

    return embedding
    
def rbf(D, params):
    # Distance radial basis function
    D_min, D_max, D_count = params["DMIN"], params["DMAX"], params["DBINS"]
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device, dtype=D.dtype)
    D_mu = D_mu[None,:]
    D_sigma = (D_max - D_min) / D_count
    RBF = torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma)**2)
    return RBF
    
def dist_to_onehot(dist, params):
    dtype = dist.dtype
    dist[torch.isnan(dist)] = 100
    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    dbins = torch.linspace(params['DMIN']+dstep, params['DMAX'], params['DBINS'], 
        dtype=torch.float, device=dist.device)
    
    db = torch.bucketize(dist.float().contiguous(), dbins).long()
    dist = torch.nn.functional.one_hot(db, num_classes=params['DBINS'] + 1)
    return dist.to(dtype)
    
def xyz_to_c6d(xyz, params, aatype, sampling=False, pseudo_beta=False):
    """convert cartesian coordinates into 2d distance 
    and orientation maps
    
    Parameters
    ----------
    xyz : pytorch tensor of shape [batch, nres, 5, 3]
          stores Cartesian coordinates of backbone N,Ca,C atoms
    Returns
    -------
    c6d : pytorch tensor of shape [batch, nres, nres, 4]
          stores stacked dist, omega, theta, phi 2D maps 
    """
    
    batch = xyz.shape[0]
    nres = xyz.shape[1]

    # three anchor atoms
    N  = xyz[:, :, 0]
    Ca = xyz[:, :, 1]
    C  = xyz[:, :, 2]
    Cb = xyz[:, :, 4] 
    if sampling: 
        if pseudo_beta:
            Cb = generate_Cbeta(N, Ca, C)
        else:
            is_gly = aatype == 7
            Cb = torch.where(
                torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
                generate_Cbeta(N, Ca, C), xyz[:, :, 4])
  
    # 6d coordinates order: (dist, omega, theta, phi)
    c6d = torch.zeros([batch, nres, nres, 4], dtype=xyz.dtype, device=xyz.device)

    dist = cdist(Cb, Cb)
    dist[torch.isnan(dist)] = 100
    c6d[..., 0] = dist + 100*torch.eye(nres, device=xyz.device, dtype=xyz.dtype)[None,...]
    b, i, j = torch.where(c6d[..., 0] < params['DMAX'])
    
    c6d[b, i, j, torch.full_like(b, 1)] = get_dih(Ca[b,i], Cb[b,i], Cb[b,j], Ca[b,j])
    c6d[b, i, j, torch.full_like(b, 2)] = get_dih(N[b,i], Ca[b,i], Cb[b,i], Cb[b,j])
    c6d[b, i, j, torch.full_like(b, 3)] = get_ang(Ca[b,i], Cb[b,i], Cb[b,j])

    # fix long-range distances
    c6d[..., 0][c6d[..., 0] >= params['DMAX']] = 100
    
    mask = torch.zeros((batch, nres, nres), dtype=xyz.dtype, device=xyz.device)
    mask[b, i, j] = 1.0
    return c6d, mask
    
def xyz_to_t2d(xyz_t, params, aatype, **kwargs):
    """convert template cartesian coordinates into 2d distance 
    and orientation maps
    
    Parameters
    ----------
    xyz_t : pytorch tensor of shape [batch, nres, 5, 3]
            stores Cartesian coordinates of template backbone N,Ca,C atoms
    params : parameters for distogram bins

    Returns
    -------
    t2d : pytorch tensor of shape [batch, nres, nres, 33 + 6 + 32]
          stores stacked dist, omega, theta, phi 2D maps 
    """
    c6d, mask = xyz_to_c6d(xyz_t, params=params, aatype=aatype, **kwargs)
    # dist to one-hot encoded
    dist = dist_to_onehot(c6d[..., 0], params)
    orien = torch.cat((torch.sin(c6d[..., 1:]), torch.cos(c6d[..., 1:])), dim=-1) * mask[..., None] # (B, L, L, 6)
    rbf_feat = rbf(cdist(xyz_t[:, :, 1], xyz_t[:, :, 1]), params) #.to(d)
    t2d = torch.cat((dist, rbf_feat, orien), dim=-1)
    t2d[torch.isnan(t2d)] = 0.0
    return t2d

# credit https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/feats.py    
def torsion_angles_to_frames(
    r: torch.Tensor,
    alpha: torch.Tensor,
    rrgdf: torch.Tensor,
) -> torch.Tensor:
    # [N, 8] transformations, i.e.
    #   One [N, 8, 3, 3] rotation matrix and
    #   One [N, 8, 3]    translation matrix
    r = Rigid.from_tensor_4x4(r)
    default_r = Rigid.from_tensor_4x4(rrgdf)
    bb_rot = torch.tensor([[0, 1]])

    # [N, 8, 2]
    alpha = torch.cat(
        [bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2
    )

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [   
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )
    all_frames_to_global = r[..., None].compose(all_frames_to_bb)
    # global backbone and sidechain frames for phi, psi, chi1, chi2, chi3, chi4
    cat_frames = Rigid.cat([r[..., None], all_frames_to_global[..., 2:]], dim=-1)
    return cat_frames.to_tensor_4x4()



def align_rigids(
    rigid_1: Rigid,
    rigid_2: Rigid,
    rigid_mask: torch.Tensor
) -> Rigid:
    """
    Aligns two `Rigid` objects using the Kabsch algorithm and a mask.

    Args:
        rigid_object_1 (Rigid): The reference rigid object.
        rigid_object_2 (Rigid): The movable rigid object to be aligned.
        rigid_mask (torch.Tensor): Boolean mask of valid elements.

    Returns:
        Rigid: Aligned `rigid_object_2`.
    """
    # Extract rotation matrices and translations
    rot1 = rigid_1.get_rots().get_rot_mats()  # Shape (*, N, 3, 3)
    trans1 = rigid_1.get_trans()  # Shape (*, N, 3)
    rot2 = rigid_2.get_rots().get_rot_mats()  
    trans2 = rigid_2.get_trans()  

    # Apply the mask
    mask = rigid_mask.unsqueeze(-1)  # Shape (*, N, 1)
    trans1_masked = trans1 * mask
    trans2_masked = trans2 * mask

    # Compute centroids for each rigid object
    mask_sums = mask.sum(dim=1)  # Shape (*, 1, 1)
    centroid1 = trans1_masked.sum(dim=1) / mask_sums  # Shape (*, 1, 3)
    centroid2 = trans2_masked.sum(dim=1) / mask_sums  # Shape (*, 1, 3)

    # Center translations
    centered_trans1 = trans1_masked - centroid1  # Shape (*, N, 3)
    centered_trans2 = trans2_masked - centroid2  # Shape (*, N, 3)

    # Compute covariance matrices
    covariances = torch.einsum("bni,bnj->bij", centered_trans1, centered_trans2)  # Shape (*, 3, 3)

    # Singular Value Decomposition (SVD)
    U, _, Vt = torch.linalg.svd(covariances)  # U, Vt: (*, 3, 3)
    optimal_rotations = torch.einsum("bij,bjk->bik", U, Vt)  # Shape (*, 3, 3)

    # Ensure right-handed coordinate system
    determinants = torch.linalg.det(optimal_rotations)  # Shape (*,)
    U[:, :, -1] *= torch.where(determinants.unsqueeze(-1) < 0, -1, 1)
    optimal_rotations = torch.einsum("bij,bjk->bik", U, Vt)

    # Apply optimal rotation to rigid_2
    rigid_2 = rigid_2.apply_trans_fn(lambda x: x - centroid2)
    return rigid_2.compose(Rigid(optimal_rotations, centroid1[:, 0]))


