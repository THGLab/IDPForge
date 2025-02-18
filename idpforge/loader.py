"""
Dataloader
created by OZ, 11/12/24
"""

import os.path
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from pytorch_lightning import LightningDataModule

from esm.esmfold.misc import collate_dense_tensors
from openfold.np.residue_constants import restype_rigid_group_default_frame, restype_order

from idpforge.misc import input_process
from idpforge.utils.diff_utils import Diffuser
from idpforge.utils.tensor_utils import get_dih, torsion_angles_to_frames


def list_index(x, indices):
    return [x[i] for i in indices]

class BatchCollator:
    def __call__(self, batch):
        # Lists to collect data from the batch
        _batch_dict = {k: [data[k] for data in batch] for k in batch[0]}

        # Apply the input_process function on the collated lists
        sequence, ss, aa_mask, residx, linker_mask = input_process(
            sequences=_batch_dict['sequence'],
            ss=_batch_dict['ss'],
            backbone_tor=[data["torsion"][:, 1:3] for data in batch], 
            residx=_batch_dict.get('resi', None)
        )

        # Map the results to a dictionary
        collated_data = {
            'sequence': sequence, 'ss': ss, 'mask': aa_mask, 'resi': residx,
        }
        for k in ['coord', 'torsion', 'frame', 'x_t', 'alpha_t', 'x_t+1', 'alpha_t+1']:
            if k in batch[0]:
                collated_data[k] = collate_dense_tensors(_batch_dict[k])
        
        if 'T' in batch[0]:
            t_list = torch.tensor(_batch_dict['T'])
            collated_data['T'] = t_list.unsqueeze(-1).repeat(1, ss.shape[1])
            
        return collated_data
    

class DiffDataset(Dataset):
    def __init__(self, ss, sequence, coords, diffuser, train=False, contact_cutoff=0):
        super(DiffDataset).__init__()
        self.template_diffuser = diffuser
        self.sequence = sequence
        self.ss = ss
        self.c = coords
        self.tstep = diffuser.T
        self.training = train
        self.contact_cutoff = contact_cutoff
    
    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        crd, R, tor = self.template_diffuser.diffuse_pose(
            self.c[idx], self.sequence[idx]) # N, CA, C, O, CB, ...
        T = -1
        if self.training:
            weights = (np.arange(self.tstep) + 1) / (np.arange(self.tstep) + 1).sum()
            T = np.random.choice(np.arange(self.tstep) + 1, size=1, p=weights)[0]
            x_t1 = torch.tensor(crd[T+1, :, :5], dtype=torch.float) if T+1 <= self.tstep else torch.tensor(crd[T, :, :5], dtype=torch.float)
            tor_t1 = torch.tensor(tor[T+1], dtype=torch.float) if T+1 <= self.tstep else torch.tensor(tor[T], dtype=torch.float)
        x_t = torch.tensor(crd[T], dtype=torch.float)
        tor_t = torch.tensor(tor[T], dtype=torch.float)
        c = torch.tensor(crd[:1], dtype=torch.float)
        
        angles = torch.zeros((len(self.ss[idx]), 7, 2), dtype=torch.float) #omega, phi, psi(-o_psi), chi1
        angles[..., 1] = -1
        angles[1:, 0] = get_dih(c[:, :-1, 1], c[:, :-1, 2], 
                        c[:, 1:, 0], c[:, 1:, 1], return_vec=True)
        angles[1:, 1] = get_dih(c[:, :-1, 2], c[:, 1:, 0], 
                    c[:, 1:, 1], c[:, 1:, 2], return_vec=True)
        angles[:, 2] = -get_dih(c[:, :, 0], c[:, :, 1], 
                    c[:, :, 2], c[:, :, 3], return_vec=True)
        angles[:, 3:] = torch.tensor(tor[0], dtype=torch.float)
        
        rrgdf = restype_rigid_group_default_frame[[restype_order[s] for s in self.sequence[idx]], ...]
        sc_frames = torsion_angles_to_frames(torch.tensor(R[0]), angles, torch.tensor(rrgdf))

        batch = {"ss": self.ss[idx], "sequence": self.sequence[idx],
                 "coord": torch.tensor(self.c[idx], dtype=torch.float),
                 "torsion": angles, # N, 7, 2
                 "frame": sc_frames, # N, 7, 4, 4
                 "x_t": x_t, # N, 5, 3
                 "alpha_t": tor_t, # N, 4, 2
                }
        if self.training:
            batch.update({"T": T - 1, "x_t+1": x_t1, "alpha_t+1": tor_t1})
        return batch
    

class IDPloader(LightningDataModule):
    def __init__(self, load_path, diffuser,
                 train_size, val_size, 
                 tr_batch_size, val_batch_size, test_batch_size, 
                 contact_cutoff=10, split_seed=42, num_workers=12):
        super().__init__()
        self.load_path = load_path
        self.train_size = train_size
        self.val_size = val_size
        self.tr_batch_size = tr_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.split_seed = split_seed
        self.nworkers = num_workers
        self.diffuser = diffuser
        self.contact_cutoff = contact_cutoff
        self.collate_fn = BatchCollator()
            
    def setup(self, stage=None):
        tr_dat = []
        val_dat = []
        test_dat = []
        
        for i, p in enumerate(self.load_path):
            with open(p, "rb") as f:
                ss, sequence, coords = pickle.load(f)
            train_size, val_size = self.train_size, self.val_size
            if isinstance(train_size, list):
                train_size = self.train_size[i]
            if isinstance(val_size, list):
                val_size = self.val_size[i]
            if val_size + train_size > 1:
                val_size = 0.99 - train_size
      
            tr, val, test = random_split(np.arange(len(ss)), [int(len(ss)*train_size), 
                int(len(ss)*val_size), len(ss) - int(len(ss)*train_size) - int(len(ss)*val_size)])
                            
            if stage == "fit" or stage is None:
                tr_dat.append(DiffDataset(list_index(ss, tr), list_index(sequence, tr), 
                        list_index(coords, tr), self.diffuser, train=True))
                val_dat.append(DiffDataset(list_index(ss, val), list_index(sequence, val), 
                        list_index(coords, val), self.diffuser)) 
            if stage == "test" or stage is None:
                test_dat.append(DiffDataset(list_index(ss, test), list_index(sequence, test), 
                        list_index(coords, test), self.diffuser))
            
        if len(tr_dat) > 1:
            self.train = ConcatDataset(tr_dat)
            self.val = ConcatDataset(val_dat)
        elif len(tr_dat) == 1:
            self.train = tr_dat[0]
            self.val = val_dat[0]
        #print(len(self.train), len(self.val))
        
        if len(test_dat) > 1:
            self.test = ConcatDataset(test_dat)
        elif len(test_dat) == 1:
            self.test = test_dat[0]
        
    def train_dataloader(self):
        return DataLoader(self.train, self.tr_batch_size, shuffle=True, 
            collate_fn=self.collate_fn, #drop_last=True, 
            pin_memory=True, num_workers=self.nworkers)
        
    def val_dataloader(self):
        return DataLoader(self.val, self.val_batch_size, 
            collate_fn=self.collate_fn,
            num_workers=self.nworkers)
        
    def test_dataloader(self):
        return DataLoader(self.test, self.test_batch_size, 
            collate_fn=self.collate_fn,
            num_workers=self.nworkers)
        
