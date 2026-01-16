# IDPForge (Intrinsically Disordered Protein, FOlded and disordered Region GEnerator)

A transformer protein language diffusion model to create all-atom IDP ensembles and IDR disordered ensembles that maintains the folded domains.

## Getting started

To get started, this repository must be cloned using the following command:

 ``` bash
 git clone https://github.com/THGLab/IDPForge.git
 ```

Following that, the working conda environment can be established in two ways.

### Installation via. IDPForge

The base environment can be built manually via the `environment.yml` file in the repo. To do this, run the following command:

``` bash
conda env create -f environment.yml

pip install -e .
```

> Note: The default file is set to install `torch==2.5.1 and cuda==12.1` for earlier GPUs (sm_60 - sm_80). Optionally, this may be changed to install `torch==2.7.1 and cuda==12.8` for later generation GPUs (sm_60 - sm_120). Refer to the comments in the file for modification instructions. 

This repo also requires OpenFold utilities, so that repository must be cloned in the same directory as IDPForge using the following command:

``` bash 
git clone https://github.com/aqlaboratory/openfold.git
```

Once the repository is cloned, proceed into the `openfold/openfold/resources` directory and run the following code: 

``` bash
wget https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
```

Once this is done, proceed back to `openfold/` and do the following:

1. Replace the `setup.py` provided with either of the `openfold_setup_12.X.py` files found in the `IDPForge/dockerfiles directory` corresponding to cuda version chosen earlier. 

2. Rename it as `setup.py` within the openfold repository.

3. Install it via `pip install -e .`. 

This makes the environment fully ready for use.

### Installation via. OpenFold

If you have issues setting up the base environment from the yml file, or if you are setting IDPForge up for use on an HPC cluster, it is recommended to follow the installation by openfold. To do this, start by cloning both repositories in the same directory.

``` bash
git clone https://github.com/THGLab/IDPForge.git

git clone https://github.com/aqlaboratory/openfold.git
```

Then proceed into `openfold/` activate the OpenFold environment using the following command:

``` bash
mamba env create -n openfold_env -f environment.yml
```

> Note: This can also be run with `conda env create -n openfold_env -f environment.yml`

Install other dependencies required by IDPForge using the following command:

``` bash
conda install einops mdtraj -c conda-forge
``` 

It is also recommended to uninstall flash-attn via `pip uninstall flash-attn` when starting out if this installation pathway is chosen.

Once flash-attn is uninstalled, proceed into `IDPForge/` and `openfold/` and install IDPForge and openfold as modules in the environment using the following commands:

``` bash
cd IDPForge/
pip install -e .
cd openfold/
pip install -e .
```
> Note: If `pip install -e .` does not work, proceed with `pip install . --no-build-isolation` instead.

This makes the environment fully ready for use.

> Note: For more information on OpenFold installation, please refer to the installation guide. https://openfold.readthedocs.io/en/latest/Installation.html

## Notes on ESM2 and Attention

ESM2 utilities are refactored into this repo for network modules and exploring the effects of ESM embedding on IDP modeling. Alternatively, it can be installed from their github https://github.com/facebookresearch/esm.git, or via pip install `pip install fair-esm`.

Optional: `pip install flash-attn==2.3` to speed up attention calculation.

## Using Docker
IDPForge can also be built as a docker container using either of the included dockerfiles (Blackwell or Ampere). Blackwell runs on CUDA12.8 and Ampere runs on CUDA12.1. Models weights and an example training data and other inference input files can be downloaded from [Figshare](https://doi.org/10.6084/m9.figshare.28414937). Optionally, the files may be merged before the creation of the image. This will ensure the image contains the merged files, removing the need for additional /weights and /data mounting.

To build the image, run the following command from the root of this repository choosing either Blackwell or Ampere based on preference:
```bash
docker build -f dockerfiles/Dockerfile_[Blackwell/Ampere] -t idpforge:latest .
```
To confirm that your idpforge:latest image is successfully completed, run
```bash
docker images
```
To run a container from the newly created image, run
```bash
docker run --rm -it --gpus all idpforge:latest
```
To verify that your docker installation is able to properly communicate with your GPU, run
```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```
Once the image is created, outside directories can be added into a container by mounting them as follows.
```bash
docker run --rm -it --gpus all \
    -v "[path-to-directory]":/app/[directory-name-in-container] \
    # Optional: any other mounts... \
    idpforge:latest
```
Examples of this are given in later sections.

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

This can also be run within the previously created docker image. Set the working directory to the root of the previously cloned and merged version of this repository and run the following.
```bash
mkdir test
sequence="GSMTPSTPPRSRGTRYLAQPSGNTSSSALMQGQKTPQKPSQNLVPVTPSTTKSFKNAPLLAPPNSNMGMTSPFNGLTSPQRSPFPKSSVKRT"
docker run -it --rm --gpus all \
    -v "./test/":/app/output \
    -v "./data/":/app/data \
    -v "./weights/":/app/weights \
    -w /app \
    idpforge:latest \
    python -u /app/sample_idp.py $sequence /app/weights/mdl.ckpt /app/output /app/configs/sample.yml --nconf 100 --cuda
```

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

This can also be run within the previously created docker image. Set the working directory to the root of the previously cloned and merged version of this repository and run the following.
```bash
mkdir P05231_build
docker run -it --rm --gpus all \
    -v "./P05231_build/":/app/output \
    -v "./data/":/app/data \
    -v "./weights/":/app/weights \
    -w /app \
    idpforge:latest \
    python -u /app/sample_ldr.py /app/weights/mdl.ckpt /app/data/AF-P05231_ndr.npz /app/output /app/configs/sample.yml --nconf 100 --cuda
```

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
