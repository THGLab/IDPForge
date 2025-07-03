# IDPForge (Intrinsically Disordered Protein, FOlded and disordered Region GEnerator)

A transformer protein language diffusion model to create all-atom IDP ensembles and IDR disordered ensembles that maintains the folded domains.

## Getting started

The environment can be built via `conda env create -f env.yml`, and optionally `pip install -e .`. This repo also requires `openfold` utilities, please refer to https://openfold.readthedocs.io/en/latest/Installation.html for installation instructions. The dependencies largely overlap with ones required by openfold. If you have issues installing from yml file, it is recommended to follow the installation by openfold, and then activate the environment and install other dependencies `conda install einops mdtraj -c conda-forge`.

ESM2 utilities are refactored into this repo for network modules and exploring the effects of ESM embedding on IDP modeling. Alternatively, it can be installed from their github https://github.com/facebookresearch/esm.git, or via pip install `pip install fair-esm`.

Optional: `pip install flash-attn==2.3` to speed up attention calculation.

### Using Docker
It can also be built as a docker container using the included dockerfile. To build it, run the following command from the root of this repository:
```bash
docker build -t idpforge .
```
To verify that your docker installation is able to properly communicate with your GPU, run
```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

Models weights and an example training data and other inference input files can be downloaded from [Figshare](https://doi.org/10.6084/m9.figshare.28414937). Unzip and move them to the corresponding folder before running scripts.

## Training

We use `pytorch-lightning` for training and one can customize training via the documented flags under `trainer` in the config file.
```bash
conda activate idpforge
python train.py --model_config_path configs/train.yml
```

## Sampling

### Single chain IDP/IDRs

We provide a commandline interface to sample single chain IDP/IDRs.
```
usage: sample_idp.py [-h] [--batch BATCH] [--nconf NCONF] [--cuda]
                     ckpt_path output_dir sample_cfg

positional arguments:
  seq                protein sequence
  ckpt_path          path to model weights
  output_dir         directory to output pdbs
  sample_cfg         path to a sampling configuration yaml file

optional arguments:
  --batch BATCH      batch size 
  --nconf NCONF      number of conformers to sample
  --cuda             whether to use cuda or cpu
```

Example to generate 100 conformers for Sic1:

```bash
mkdir test
sequence="GSMTPSTPPRSRGTRYLAQPSGNTSSSALMQGQKTPQKPSQNLVPVTPSTTKSFKNAPLLAPPNSNMGMTSPFNGLTSPQRSPFPKSSVKRT"
python sample_idp.py $sequence weights/mdl.ckpt test configs/sample.yml --nconf 100 --cuda 
```

Inference time experimental guidance can be activated by the potential flag in the `configs/sample.yml`. An example PREs experimental data file is also provided in `data/sic1_pre_exp.txt`.

### IDRs with folded domains

First, to prepare the folded template, run `python init_ldr_template.py`. We provide an example for sampling the low confidence region of AF entry P05231:
```bash
python mk_ldr_template.py data/AF-P05231-F1-model_v4.pdb 1-41 data/AF-P05231_ndr.npz
```
The provided model weights are not recommended for predicting multiple domains at the same time.

Then, to generate an IDRs with folded domains ensemble, run
```bash
mkdir P05231_build
python sample_ldr.py weights/mdl.ckpt data/AF-P05231_ndr.npz P05231_build configs/sample.yml --nconf 100 --cuda
```
One can set the `attention_chunk` to manage memory usage for long sequences (Inference on long disordered sequences may be limited by training sequence length).

### Chemical shifts prediction and evaluating ensembles with X-EISD (optional)

We use UCBShift for chemical shift prediction and can be installed at https://github.com/THGLab/CSpred.git. If you wish to use X-EISD for evaluation or reweighing with experimental data, please refer to https://github.com/THGLab/X-EISDv2.

## Citation
```bibtex
@article{zhang2025,
  author    = {Zhang, Oufan and Liu, Zi-Hao and Forman-Kay, Julie D. Head-Gordon, Teresa},
  title     = {Deep Learning of Proteins with Local and Global Regions of Disorder},
  journal   = {arXiv preprint},
  year      = {2025},
  archivePrefix = {arXiv},
  eprint    = {2502.11326},
}
```
