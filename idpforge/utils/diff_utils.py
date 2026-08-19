"""
Adapted from https://github.com/RosettaCommons/RFdiffusion.git

Copyright (c) 2023 University of Washington. Developed at the Institute for
Protein Design by Joseph Watson, David Juergens, Nathaniel Bennett, Brian Trippe
and Jason Yim. This copyright and license covers both the source code and model
weights referenced for download in the README file.
"""

import torch 
import numpy as np
import pickle
import os
from scipy.spatial.transform import Rotation as scipy_R

from idpforge.utils.igso3_utils import Exp, calculate_igso3
from idpforge.utils.np_utils import get_chi_angles, rigid_from_3_points_np
from idpforge.utils.definitions import backbone_atom_positions


def wrap_rad(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def align_coords(crd1, crd2, atom_mask):
    """Aligns two sets of batched protein Cα atom coordinates using the Kabsch algorithm."""
    # Mask coordinates (B, N, 3)
    crd1_masked = crd1[..., 1, :] * atom_mask[..., None]
    crd2_masked = crd2[..., 1, :] * atom_mask[..., None]
    mask_sums = atom_mask.sum(axis=1, keepdims=True)  
    centroid1 = crd1_masked.sum(axis=1) / mask_sums  # (B, 3)
    centroid2 = crd2_masked.sum(axis=1) / mask_sums  # (B, 3)

    # Center coordinates
    centered1 = crd1_masked - centroid1[:, None, :]
    centered2 = crd2_masked - centroid2[:, None, :]
    covariances = np.einsum("bni,bnj->bij", centered1, centered2)
    try:
        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(covariances)  # U, Vt: (B, 3, 3)
        align_rots = np.einsum("bij,bjk->bik", U, Vt)  # (B, 3, 3)

        # Ensure right-handed coordinate systems
        determinants = np.linalg.det(align_rots)  # (B,)
        U[:, :, -1] *= np.where(determinants < 0, -1, 1)[:, None]
        align_rots = np.einsum("bij,bjk->bik", U, Vt)
        R = scipy_R.from_matrix(align_rots).as_matrix().reshape(-1, 3, 3)    
        rotated = np.einsum("bij,bnmj->bnmi", R, crd2 - centroid2[:, None, None, :])
    except np.linalg.LinAlgError:
        print("SVG not converged; rotation not applied")
        rotated = crd2 - centroid2[:, None, None, :]

    return rotated + centroid1[:, None, None, :]


def linear_beta_schedule(T, b0, bT, steps=200): 
    """Given a noise schedule type, create the beta schedule."""
    b0 *= T / steps 
    bT *= T / steps
    schedule = np.linspace(b0, bT, steps + 1) 
    return schedule

    
def get_noise_schedule(T, noiseT, noise1, schedule_type="linear"):
    """Create a schedule that varies the noise scale over time."""

    noise_schedules = {
        "constant": lambda t: noiseT,
        "linear": lambda t: (t / T) * (noiseT - noise1) + noise1,
    }

    assert (
        schedule_type in noise_schedules
    ), f"noise_schedule must be one of {noise_schedules.keys()}. Received noise_schedule={schedule_type}. Exiting."

    return noise_schedules[schedule_type]
    
def get_mu_xt_x0(xt, px0, t, beta_schedule, alphabar_schedule, eps=1e-6):
    """Given xt, predicted x0 and timestep t, give mu of x(t-1)."""
    sigma = (
        (1 - alphabar_schedule[t - 2]) / (1 - alphabar_schedule[t - 1])
    ) * beta_schedule[t - 1]
    
    xt_ca = xt[..., 1, :]
    px0_ca = px0[..., 1, :]

    a = ((np.sqrt(alphabar_schedule[t - 2] + eps) * beta_schedule[t - 1])
        / (1 - alphabar_schedule[t - 1])) * px0_ca
        
    b = ((np.sqrt(1 - beta_schedule[t - 1] + eps)
            * (1 - alphabar_schedule[t - 2])) / (1 - alphabar_schedule[t - 1])
    ) * xt_ca

    mu = a + b

    return mu, sigma

def get_next_ca(
    xt,
    px0,
    t,
    diffusion_mask,
    crd_scale,
    beta_schedule,
    alphabar_schedule,
    noise_scale=1.0,
):
    """Given full-atom x0 prediction (coordinates), diffuse to x(t-1)."""
    L = len(xt)
    px0 = px0 * crd_scale
    xt = xt * crd_scale

    mu, sigma = get_mu_xt_x0(
        xt, px0, t, beta_schedule=beta_schedule, alphabar_schedule=alphabar_schedule
    )
    sampled_crds = np.random.normal(mu, np.sqrt(sigma * noise_scale))
    delta = sampled_crds - xt[..., 1, :]

    if not diffusion_mask is None:
        # Don't move motif
        delta[~diffusion_mask, ...] = 0

    out_crds = xt + delta[..., None, :]
    return delta / crd_scale

def get_next_frames(xt, px0, t, diffuser, diffusion_mask, noise_scale=1.0):
    """Get updated frames at t-1 using IGSO(3) score-based reverse diffusion."""
    B, L = xt.shape[:2]
    R_0 = rigid_from_3_points_np(px0).reshape(-1, 3, 3)
    R_t = rigid_from_3_points_np(xt).reshape(-1, 3, 3)
    Ca_t = xt[..., 1, :]

    # Replace degenerate rotation matrices (det ~ 0) with identity.
    for R in [R_0, R_t]:
        dets = np.linalg.det(R)
        bad = (np.abs(dets) < 1e-6) | np.isnan(dets)
        if np.any(bad):
            R[bad] = np.eye(3)

    try:
        R_0 = scipy_R.from_matrix(R_0).as_matrix().reshape(B, L, 3, 3)
        R_t = scipy_R.from_matrix(R_t).as_matrix().reshape(B, L, 3, 3)
    except np.linalg.LinAlgError:
        print("SVD not converged in get_next_frames; rotation not applied")
        return xt.copy(), np.tile(np.identity(3), (B, L, 1, 1))
    all_rot_transitions = np.tile(np.identity(3), (B, L, 1, 1))
  
    # Sample next frame for each residue
    dr = diffuser.so3_diffuser.reverse_sample_vectorized(
        R_t, R_0, t,
        noise_level=noise_scale,
    )
    all_rot_transitions[diffusion_mask, :, :] = dr[diffusion_mask, :, :]

    # Apply the interpolated rotation matrices to the coordinates
    next_crds = np.einsum("...ij,...aj->...ai", all_rot_transitions, 
                          xt - Ca_t[..., None, :]) + Ca_t[..., None, :]

    # (B, L, 5, 3) set of backbone coordinates with slight rotation, rot_matrix (B, L, 3, 3)
    return next_crds, all_rot_transitions

def get_next_chi_angles(
    xt,
    px0,
    t,
    diffusion_mask,
    beta_schedule,
    alphabar_schedule,
    noise_scale=1.0,
    eps=1e-6
):
    """Given full-atom x0 prediction (sidechain torsions), diffuse to x(t-1)."""
    sigma = (
        (1 - alphabar_schedule[t - 2]) / (1 - alphabar_schedule[t - 1])
    ) * beta_schedule[t - 1]
    mu = ((np.sqrt(alphabar_schedule[t - 2] + eps) * beta_schedule[t - 1]) 
          / (1 - alphabar_schedule[t - 1])) * px0 + ((np.sqrt(1 - beta_schedule[t - 1] + eps)
            * (1 - alphabar_schedule[t - 2])) / (1 - alphabar_schedule[t - 1])) * xt

    sampled_tors = np.mod(np.random.normal(mu, np.sqrt(sigma * noise_scale)) + np.pi, 2 * np.pi) - np.pi

    if not diffusion_mask is None and np.any(diffusion_mask == 0):
        sampled_tors[~diffusion_mask] = xt[~diffusion_mask]

    return sampled_tors
    
def init_sample(sequence, T, so3_diffuser, crd_scale=0.25):
    # initiate at origin
    xyz = np.array([backbone_atom_positions[s] for s in sequence])
    sampled_trans = np.random.normal(size=(len(sequence), 3)) / crd_scale
    
    # Sample rotations and scores from IGSO3
    sampled_rots = so3_diffuser.sample_vec(np.array([T]), n_samples=len(sequence))  # [T, N, 3]
    
    # Apply sampled rot.
    R_sampled = scipy_R.from_rotvec(sampled_rots.reshape(-1, 3)).as_matrix().reshape(len(sequence), 3, 3)
    xyz = np.einsum("nij,naj->nai", R_sampled, xyz) + sampled_trans[:, None, :]
    
    # Sample torsion angles in [-pi, pi)
    sampled_torsions = wrap_rad(np.random.normal(size=(len(sequence), 4)))
    sampled_alphas = np.stack((np.sin(sampled_torsions), np.cos(sampled_torsions)), axis=-1)
    return xyz, sampled_alphas 
        
    
class EuclideanDiffuser:
    # class for diffusing points in 3D

    def __init__(self, T, b_0=0.01, b_T=0.08): 
        self.T = T
        self.schedule_param = (b_0, b_T)
        self.beta_schedule = linear_beta_schedule(T, b_0, b_T)

    def diffuse_translations(self, xyz, diffusion_mask=None, var_scale=1):
        return self.apply_kernel_recursive(xyz, diffusion_mask, var_scale)

    def apply_kernel(self, x, t, diffusion_mask=None, var_scale=1):
        """Applies a noising kernel to the points in x."""
        assert len(x.shape) == 3
        L, _, _ = x.shape

        # c-alpha crds
        ca_xyz = x[:, 1, :]
      
        b_t = self.beta_schedule[t]
   
        # get the noise at timestep t
        mean = (np.sqrt(1 - b_t) * ca_xyz)
        var = (np.ones((L, 3)) * (b_t) * var_scale)
        
        sampled_crds = np.random.normal(mean, np.sqrt(var))
        delta = sampled_crds - ca_xyz

        if not diffusion_mask is None or np.any(diffusion_mask == 0):
            delta[~diffusion_mask] = 0

        out_crds = x + delta[:, None, :]

        return out_crds, delta

    def apply_kernel_recursive(self, xyz, diffusion_mask=None, var_scale=1):
        """Repeatedly apply self.apply_kernel T times and return all crds."""
        
        bb_stack = [xyz.copy()]
        T_stack = [np.zeros_like(xyz[:, 1])]

        for t in range(self.T):
            xyz, cur_T = self.apply_kernel(
                xyz, t, var_scale=var_scale, 
                diffusion_mask=diffusion_mask
            )
            bb_stack.append(xyz.copy())
            T_stack.append(cur_T.copy())

        return np.array(bb_stack), np.array(T_stack)


class TorsionDiffuser:
    # class for diffusing torsions

    def __init__(self, T, b_0=0.01, b_T=0.06):
        self.T = T
        self.schedule_param = (b_0, b_T)
        self.beta_schedule = linear_beta_schedule(T, b_0, b_T)

    def apply_kernel(self, x, t, diffusion_mask=None, var_scale=1, exists_mask=None):
        """Applies a noising kernel to the points in x."""
        L, _ = x.shape
        b_t = self.beta_schedule[t]
   
        # get the noise at timestep t
        mean = np.sqrt(1 - b_t) * x
        var = np.ones((L, 4)) * b_t * var_scale
        sampled_torsions = np.mod(np.random.normal(mean, np.sqrt(var)) + np.pi, 2 * np.pi) - np.pi
        
        if diffusion_mask is not None:
            assert diffusion_mask.dtype == np.bool_
            sampled_torsions[~diffusion_mask] = x[~diffusion_mask]
        if exists_mask is not None:
            assert exists_mask.dtype == np.bool_
            sampled_torsions[~exists_mask] = -np.pi

        return sampled_torsions

    def diffuse_torsions(self, torsions, diffusion_mask=None, var_scale=1, exists_mask=None):
        """Repeatedly apply self.apply_kernel T times and return all torsions."""
        
        stack = [torsions.copy()]
        sampled_tor = torsions.copy()

        for t in range(self.T):
            sampled_tor = self.apply_kernel(
                sampled_tor, t, 
                var_scale=var_scale, 
                diffusion_mask=diffusion_mask,
                exists_mask=exists_mask
            )
            stack.append(sampled_tor)

        return np.array(stack)



class IGSO3:
    """Perform IGSO3 diffusion on a set of backbone coordinates."""

    def __init__(
        self,
        T,
        min_sigma=0.05,
        min_b=1.5,
        max_b=2.5, 
        cache_file=None,
        num_omega=1000,
    ):
        self.T = T

        self.min_b = min_b
        self.max_b = max_b
        self.min_sigma = min_sigma
        self.max_sigma = self.sigma(1.0)
        self.num_omega = num_omega
        self.num_sigma = 500

        # Calculate igso3 values.
        if cache_file is not None and os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.igso3_vals = pickle.load(f)
        else: 
            self.igso3_vals = calculate_igso3(
                num_sigma=self.num_sigma,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                num_omega=self.num_omega
            )
        self.step_size = 1 / self.T

    def save_igso3(self, cache_file):
        with open(cache_file, "wb") as handle:
            pickle.dump(self.igso3_vals, handle)

    @property
    def discrete_sigma(self):
        return self.igso3_vals["discrete_sigma"]

    def sigma_idx(self, sigma: np.ndarray):
        """Calculates the index for discretized sigma during IGSO(3) initialization."""
        return np.digitize(sigma, self.discrete_sigma) - 1

    def t_to_idx(self, t: np.ndarray):
        """Go from discrete time index t to corresponding sigma_idx."""
        continuous_t = t / self.T
        return self.sigma_idx(self.sigma(continuous_t))

    def sigma(self, t: torch.Tensor):
        r"""Extract \sigma(t) corresponding to chosen sigma schedule."""
        t_type = "torch"
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
            t_type = "numpy"
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f"Invalid t={t.item()}")
            
        x = self.min_sigma + t * self.min_b + (t**2) * (self.max_b - self.min_b) / 2
        return x if t_type == "torch" else x.numpy()

    def g(self, t):
        r"""Return the drift coefficient at time t."""
        with torch.enable_grad():
            t = torch.tensor(t, requires_grad=True)
            sigma_sqr = self.sigma(t) ** 2
            grads = torch.autograd.grad(sigma_sqr.sum(), t)[0]
        return np.sqrt(grads.detach().numpy())

    def sample(self, ts, n_samples=1):
        """Sample an angle of rotation from IGSO(3) using the inverse cdf."""
        assert sum(ts == 0) == 0, "assumes one-indexed, not zero indexed"
        all_samples = []
        for t in ts:
            sigma_idx = self.t_to_idx(t)
            sample_i = np.interp(
                np.random.rand(n_samples),
                self.igso3_vals["cdf"][sigma_idx],
                self.igso3_vals["discrete_omega"],
            )  # [N, 1]
            all_samples.append(sample_i)
        return np.stack(all_samples, axis=0)

    def sample_vec(self, ts, n_samples=1):
        """Generate rotation vector(s) from IGSO(3) at time steps ts."""
        x = np.random.randn(len(ts), n_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return x * self.sample(ts, n_samples=n_samples)[..., None]

    def score_norm(self, t, omega):
        """Compute the score norm based on the time step and angle."""
        sigma_idx = self.t_to_idx(t)
        score_norm_t = np.interp(
            omega,
            self.igso3_vals["discrete_omega"],
            self.igso3_vals["score_norm"][sigma_idx],
        )
        return score_norm_t

    def score_vec(self, ts, vec):
        """Compute the score of the IGSO(3) density as a rotation vector."""
        omega = np.linalg.norm(vec, axis=-1)
        all_score_norm = []
        for i, t in enumerate(ts):
            omega_t = omega[i]
            t_idx = t - 1
            sigma_idx = self.t_to_idx(t)
            score_norm_t = np.interp(
                omega_t,
                self.igso3_vals["discrete_omega"],
                self.igso3_vals["score_norm"][sigma_idx],
            )[:, None]
            all_score_norm.append(score_norm_t)
        score_norm = np.stack(all_score_norm, axis=0)
        return score_norm * vec / omega[..., None]

    def exp_score_norm(self, ts):
        """Return the expected value of the norm of the score for IGSO(3)."""
        sigma_idcs = [self.t_to_idx(t) for t in ts]
        return self.igso3_vals["exp_score_norms"][sigma_idcs]

    def diffuse_frames(self, xyz, diffusion_mask=None):
        """Sample from the IGSO(3) distribution to noise frames."""
        t = np.arange(self.T) + 1
        num_res = len(xyz)
            
        # scipy rotation object for true coordinates
        R_true = rigid_from_3_points_np(xyz[None, ...])
        Ca = xyz[:, 1, :]

        # Sample rotations and scores from IGSO3
        sampled_rots = self.sample_vec(t, n_samples=num_res)  # [T, N, 3]

        if diffusion_mask is not None or np.any(diffusion_mask == 0):
            sampled_rots = sampled_rots * diffusion_mask[None, :, None]

        # Apply sampled rot.
        R_sampled = scipy_R.from_rotvec(sampled_rots.reshape(-1, 3)).as_matrix().reshape(self.T, num_res, 3, 3)
        R_perturbed = np.einsum("tnij,njk->tnik", R_sampled, R_true[0]) 
        perturbed_crds = np.einsum("tnij,naj->tnai", R_sampled, xyz - Ca[:, None, :]
            ) + Ca[None, :, None, :]
    
        # append the starting xyz and frames
        perturbed_crds = np.concatenate((xyz[None, ...], perturbed_crds), axis=0)
        R_perturbed = np.concatenate((R_true, R_perturbed), axis=0)

        return perturbed_crds, R_perturbed


    def reverse_sample_vectorized(self, R_t, R_0, t, noise_level, eps=1e-6):
        """Sample a rotation at the previous time step using an IGSO3 score approximation."""
        # rotation vector from r_t to r_0
        B, L = R_0.shape[:2]
        R_0t = np.einsum("...ij,...kj->...ik", R_t, R_0).reshape(-1, 3, 3)
        # Sanitize degenerate matrices before SVD
        bad = np.isnan(R_0t).any(axis=(-2, -1))
        if np.any(bad):
            R_0t[bad] = np.eye(3)
        try:
            R_0t_rotvec = scipy_R.from_matrix(R_0t).as_rotvec() # shape *, 3
        except np.linalg.LinAlgError:
            print("SVD not converged in reverse_sample; rotation not applied")
            return np.tile(np.identity(3), (B, L, 1, 1))
     
        # Approximate the score based on the prediction of R0.
        Omega = np.linalg.norm(R_0t_rotvec, axis=-1)
        Omega[Omega == 0] = eps
        Score_approx = R_0t_rotvec * (self.score_norm(t, Omega) / Omega)[..., None]

        # Compute scaling for score and sampled noise
        continuous_t = t / self.T
        rot_g = self.g(continuous_t)

        # Sample and scale noise in the SO(3) tangent space

        Z = np.random.normal(np.zeros((B * L, 3)), 1)
        Z *= noise_level

        Delta_r = (rot_g**2) * self.step_size * Score_approx
    
        # Sample perturbation from the discretized SDE
        Perturb_tangent = Delta_r + rot_g * np.sqrt(self.step_size) * Z # B * L, 3
        Perturb = Exp(Perturb_tangent).reshape(B, L, 3, 3)
        return Perturb


class Diffuser:
    def __init__(
        self,
        T,
        crd_scale=0.25,
        var_scale=1.,
        euclid_b0=0.01,
        euclid_bT=0.06,
        tor_b0=0.01,
        tor_bT=0.06,
        cache=None
    ):
        self.T = T
        self.crd_scale = crd_scale
        self.var_scale = var_scale

        # get backbone frame diffuser
        self.so3_diffuser = IGSO3(
            T=self.T,
            cache_file=cache
        )

        # get backbone translation diffuser
        self.eucl_diffuser = EuclideanDiffuser(self.T, euclid_b0, euclid_bT)
        self.tor_diffuser = TorsionDiffuser(self.T, tor_b0, tor_bT)

        print("Successful diffuser __init__")

    def diffuse_pose(
        self,
        xyz,
        sequence,
        diffusion_mask=None,
    ):
        """Given full-atom xyz and sequence, diffuse the protein frame translations and rotations."""
        
        L = len(xyz)
        assert not np.isnan(xyz).any()
        xyz = np.array(xyz, dtype=float, copy=True)

        # center the held structure at origin
        if diffusion_mask is not None:
            assert diffusion_mask.dtype == np.bool_
            held = ~diffusion_mask
            com = xyz[:, 1][held].mean(axis=0) if held.any() else xyz[:, 1].mean(axis=0)
        else:
            com = xyz[:, 1].mean(axis=0)
        xyz -= com

        # 1 get frames
        diffused_frame_crds, R_deltas = self.so3_diffuser.diffuse_frames(
            xyz[:, :5], diffusion_mask=diffusion_mask
        )

        # 2 get translations
        _, deltas = self.eucl_diffuser.diffuse_translations(
            xyz[:, :5] * self.crd_scale, diffusion_mask=diffusion_mask, var_scale=self.var_scale
        )
        deltas /= self.crd_scale

        # 3 get torsions
        torsions, torsion_mask = get_chi_angles(xyz, sequence)
        diffused_torsions = self.tor_diffuser.diffuse_torsions(
            torsions,
            diffusion_mask=None if diffusion_mask is None else np.repeat(diffusion_mask[:, None], 4, 1),
            var_scale=self.var_scale,
            exists_mask=torsion_mask
        )
        diffused_torsions = np.stack((np.sin(diffused_torsions), np.cos(diffused_torsions)), axis=-1)

        # Now combine all the diffused quantities to make full atom diffused poses
        cum_deltas = np.cumsum(deltas, axis=0)
        diffused_BB = diffused_frame_crds + cum_deltas[:, :, None, :]  # [T, L, 3, 3] 
        
        rigids = np.zeros((self.T + 1, L, 4, 4))
        rigids[..., :3, :3] = R_deltas
        rigids[..., :3, 3] = diffused_BB[:, :, 1]
        rigids[..., 3, 3] = 1
            
        return diffused_BB, rigids, diffused_torsions # [T, L, 5, 3], [T, L, 4, 4], [T, L, 4, 2]



class Denoiser:
    """Get x(t-1) from predicted x0 and x(t)."""

    def __init__(
        self,
        ntsteps,
        diffuser,
        noise_scale_ca=0.2,
        final_noise_scale_ca=1.,
        noise_scale_frame=1,
        final_noise_scale_frame=0,
    ):
        self.T = ntsteps
        self.diffuser = diffuser
        self.crd_scale = diffuser.crd_scale

        self.noise_schedule_ca = get_noise_schedule(
            diffuser.T,
            final_noise_scale_ca,
            noise_scale_ca,
            "constant" if final_noise_scale_ca == 0 else "linear"
        )
        self.noise_schedule_frame = get_noise_schedule(
            diffuser.T,
            final_noise_scale_frame,
            noise_scale_frame,
            "constant" if final_noise_scale_frame == 0 else "linear"
        )
        self.trans_betabar = linear_beta_schedule(diffuser.T, diffuser.eucl_diffuser.schedule_param[0], 
                diffuser.eucl_diffuser.schedule_param[1], ntsteps)
        self.tor_betabar = linear_beta_schedule(diffuser.T, diffuser.tor_diffuser.schedule_param[0], 
                diffuser.tor_diffuser.schedule_param[1], ntsteps)
        self.trans_alphabar = np.cumprod(1 - self.trans_betabar, axis=0)
        self.tor_alphabar = np.cumprod(1 - self.tor_betabar, axis=0)
        
    def init_samples(self, sequences, crd_scale=None, device=torch.device("cpu")):
        xyzs = []
        tors = []
        if crd_scale is None:
            crd_scale = self.diffuser.crd_scale
            
        for i, s in enumerate(sequences):
            xyz, torsion = init_sample(s, self.diffuser.T, self.diffuser.so3_diffuser, crd_scale)
            xyzs.append(torch.tensor(xyz, device=device, dtype=torch.float))
            tors.append(torch.tensor(torsion, device=device, dtype=torch.float))
        return xyzs, tors


    def get_next_pose(
        self,
        x_t,
        px_0,
        tor_t, 
        ptor_0,
        t,
        torsion_mask,
        diffusion_mask,
        motiff_mask=None,
    ):
        """Take px0, xt and t and produce xt-1."""

        L, n_atom = x_t.shape[:2]
        # align px0 to xt
        if motiff_mask is not None:
            px_0 = align_coords(x_t, px_0, motiff_mask)  
            diffusion_mask = diffusion_mask & ~motiff_mask
        else:
            px_0 = align_coords(x_t, px_0, diffusion_mask)

        # get the next set of CA coordinates
        noise_scale_ca = self.noise_schedule_ca(t)
        ca_deltas = get_next_ca(
            x_t,
            px_0,
            int(t * self.T / self.diffuser.T),
            diffusion_mask,
            crd_scale=self.crd_scale,
            beta_schedule=self.trans_betabar,
            alphabar_schedule=self.trans_alphabar,
            noise_scale=noise_scale_ca,
        )

        # get the next set of backbone frames (coordinates)
        noise_scale_frame = self.noise_schedule_frame(t) 
        crds_next, frames = get_next_frames(
            x_t,
            px_0,
            t,
            self.diffuser,
            diffusion_mask,
            noise_scale=noise_scale_frame,
        )
        # get the next set of torsions
        tor_t = np.arctan2(tor_t[..., 0], tor_t[..., 1])
        ptor_0 = np.arctan2(ptor_0[..., 0], ptor_0[..., 1])
        torsion_diffusion_mask = torsion_mask if diffusion_mask is None else torsion_mask & diffusion_mask[..., None]
        torsion_next = get_next_chi_angles(
            tor_t,
            ptor_0,
            int(t * self.T / self.diffuser.T),
            torsion_diffusion_mask,
            self.tor_betabar,
            self.tor_alphabar,
            noise_scale_ca
        )
        torsion_next = np.stack((np.sin(torsion_next), np.cos(torsion_next)), axis=-1)
 
        # add the delta to the new frames
        crds_next += ca_deltas[..., None, :]  # translate
        
        return crds_next, torsion_next
