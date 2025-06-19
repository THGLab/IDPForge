## Documentation

Relavent data files can be downloaded from https://doi.org/10.6084/m9.figshare.28414937.

`train_idr_seqs.fasta`: IDP/IDRs training sequences from DisProt and IDRome (https://github.com/KULL-Centre/_2023_Tesei_IDRome.git).

`idpcg_disprot`: IDPConformerGenerator (https://github.com/julie-forman-kay-lab/IDPConformerGenerator.git) generated DisProt conformers.

`example_data.pkl`: an example training data pickle. It contains lists of secondary structure encodings, amino sequences and heavy atom coordinates. An input pdb can be prepared by

```python
import numpy as np
import mdtraj as md
from idpforge.utils.np_utils import (
    process_pdb, assign_rama
)

def parse_pdb(pdb):
    crd, seq = process_pdb(pdb)
    traj = md.load(pdb)
    dssp = md.compute_dssp(traj, simplified=True)[0]
    phis = md.compute_phi(traj)[1][0]
    psis = md.compute_psi(traj)[1][0]
    phis = np.concatenate(([-np.pi], phis))
    psis = np.concatenate((psis, [np.pi]))
    rama = assign_rama(np.stack([phis, psis], axis=-1))
    encode = "".join([dssp[i] if dssp[i] in ["H", "E"] else rama[i] for i in range(len(dssp))])
    return encode, seq, crd

sec, seq, crd = parse_pdb("input.pdb")
```

`diff_igso3.pkl`: cached IGSO3 discretization for 200 timesteps on a linear schedule; will generate based on diffusion schedule parameters if not provided.

`sic1_pre_exp.txt`: Sic1 PRE data file (in distance-based representation).

`example_sec.txt`: compiled secondary structure encoding examples; can be customized based on sampling needs (for example enforcing residues to be helices or sheets). Setting the `data_path` but deactivating `sec_path` in the config file (by default) will generate secondary structure encoding based on sequence preference based on the compiled pdb data, or can be manually generated from:
```bash
cd ../idpforge/utils
python prep_sec.py --sequence <AAA> --database <PATH_TO_DATA.pkl> --nsample 500 --output <PATH_TO_SEC.txt>
```

