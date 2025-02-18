"""
Adapted from https://github.com/RosettaCommons/RFdiffusion.git

Copyright (c) 2023 University of Washington. Developed at the Institute for
Protein Design by Joseph Watson, David Juergens, Nathaniel Bennett, Brian Trippe
and Jason Yim. This copyright and license covers both the source code and model
weights referenced for download in the README file.
"""

import torch
import math
from idpforge.utils.tensor_utils import get_dih


class Potential:
    '''
    Interface class that defines the functions a potential must implement
    '''
    def compute(self, xyz, **kwargs):
        '''
        Given the current structure of the model prediction, return the current
        potential as a PyTorch tensor with a single entry

        Args:
            xyz (torch.tensor, size: [B, L, ...]: The current coordinates of the sample
            
        Returns:
            potential (torch.tensor, size: [B]): A potential whose value will be MAXIMIZED
                                                     by taking a step along it's gradient
        '''
        raise NotImplementedError('Potential compute function was not overwritten')
    
    def get_potential_gradients(self, xyz):
        '''
        calculates grads on CA only for distance based potentials;
        calculates grads on backbones(N-Ca-C) for JCouplings; O & Cb gradients may not be correct thus should not be used as input features
        '''
        with torch.enable_grad():
            xyz.requires_grad = True
            if xyz.grad is not None:
                xyz.grad.zero_()
    
            current_potential = self.compute(xyz)
            current_potential.backward()
        grads = xyz.grad
        if torch.sum(grads[:, :, 2: 5]) == 0:
            grads = torch.tile(grads[:, :, 1][..., None, :], (1, 1, 5, 1))
            assert grads.shape == xyz.grad.shape
        if torch.isnan(grads[:, :, 1]).any():
            print("WARNING: NaN detected in Ca and replaced with zero")
            grads[:, 0] = 0
            grads[torch.isnan(grads)] = 0
        return grads
        
        
class RoG(Potential):

    def __init__(self, target, **kwargs):
        self.target = target

    def compute(self, xyz):
        Ca = xyz[:, :, 1] # B, L, 3
        centroid = torch.mean(Ca, dim=1) # shape B, 3
        dgram = torch.cdist(Ca.contiguous(), centroid[:, None, :].contiguous(), p=2).squeeze(-1) # [B, L]
        rg = torch.sqrt(torch.sum(dgram ** 2, dim=-1)) # [B, ]
        loss = (rg.mean() - self.target) ** 2 / 2.
        return -loss


class Contact(Potential):
    '''
    Differentiable way to maximise number of contacts within a protein
    Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html
    NOTE: This function sometimes produces NaN's -- added check in reverse diffusion for nan grads
    '''

    def __init__(self, contact_bounds, **kwargs):

        self.bound = torch.tensor(contact_bounds, dtype=torch.float)
        self.mask = torch.tensor(contact_bounds[..., 0] > 0)
        self.rand_mask_prob = kwargs.get("exp_mask_p", 1)
        self.eps = 1e-3

    def to(self, device):
        self.bound = self.bound.to(device)
        self.mask = self.mask.to(device)

    def compute(self, xyz):
        assert len(self.mask) == xyz.shape[1], "structure shape inconsistent with exp data"
        if xyz.device != self.bound.device:
            self.to(xyz.device)
        Ca = xyz[:, :, 1] # [B, L, 3]
        dgram = torch.cdist(Ca.contiguous(), Ca.contiguous(), p=2)
        avg_dgram = torch.pow(torch.pow(dgram + self.eps, -6).mean(dim=0), -1/6) 
        loss = torch.clamp(self.bound[..., 0] - avg_dgram, min=0)**2 + torch.clamp(avg_dgram - self.bound[..., 1], min=0)**2
        rand_mask = torch.rand_like(self.mask.float()) < self.rand_mask_prob
        contacts = loss * self.mask * rand_mask / 2. # [L, L]
        return -contacts.sum() #/ self.mask.sum()


class Efret(Potential):
    def __init__(self, exp_val, **kwargs):
        self.target = torch.tensor(exp_val[..., 0], dtype=torch.float)
        self.scaler = torch.tensor(exp_val[..., 1], dtype=torch.float)
        self.mask = self.target > 0
        self.rand_mask_prob = kwargs.get("exp_mask_p", 1)
        self.eps = 1e-3

    def to(self, device):
        self.target = self.target.to(device)
        self.scaler = self.scaler.to(device)
        self.mask = self.mask.to(device)

    def compute(self, xyz):
        assert len(self.mask) == xyz.shape[1], "structure shape inconsistent with exp data"
        if xyz.device != self.target.device:
            self.to(xyz.device)
        Ca = xyz[:, :, 1] # [B, L, 3]
        dgram = torch.cdist(Ca.contiguous(), Ca.contiguous(), p=2)
        eff = 1.0 / (1.0 + (dgram / self.scaler) ** 6)
        loss = torch.abs(torch.mean(eff, dim=0) - self.target) 
        rand_mask = torch.rand_like(self.mask.float()) < self.rand_mask_prob
        return -(loss * self.mask * rand_mask).sum() #/ self.mask.sum()


class JCoup(Potential):

    def __init__(self, exp_val, **kwargs):
        self.target = torch.tensor(exp_val, dtype=torch.float)
        self.mask = self.target > 0
        self.rand_mask_prob = kwargs.get("exp_mask_p", 1)

    def to(self, device):
        self.target = self.target.to(device)
        self.mask = self.mask.to(device)

    def compute(self, xyz):
        assert len(self.mask) == xyz.shape[1] - 1, "structure shape inconsistent with exp data"
        # conform data and structure device
        if xyz.device != self.target.device:
            self.to(xyz.device)
        # cos (phi - 60 deg)
        phi = get_dih(xyz[:, :-1, 2], xyz[:, 1:, 0], xyz[:, 1:, 1], xyz[:, 1:, 2]) # [B, L]
        alpha = torch.cos(phi - 2 * math.pi / 3) #phi[:, :, 0] * 0.5 + phi[:, :, 1] * math.sin(2 * math.pi / 3) 
        jc = torch.mean(6.51 * alpha ** 2 - 1.76 * alpha + 1.6, dim=0)
        rand_mask = torch.rand_like(self.mask.float()) < self.rand_mask_prob
        loss = (jc - self.target) ** 2 * self.mask * rand_mask / 2.
        return -loss.sum() #/ mask.sum()


class Multiple(Potential):
    
    def __init__(self, configs, weights, **kwargs):
        self.terms = [Potentials[k](**cfg) for k, cfg in configs.items()]
        self.weights = torch.tensor([weights[k] for k in configs])
    
    def compute(self, xyz):
        potential = torch.stack([w * term.compute(xyz) for w, term in zip(self.weights, self.terms)])
        return torch.sum(potential) / sum(self.weights)
        
        
Potentials = {
    "rg": RoG,
    "contact": Contact,
    "jcoup": JCoup,
    "fret": Efret, 
    "multiple": Multiple,
}
        
