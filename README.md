# IDPForge (Intrinsically Disordered Protein, FOlded and disordered Region GEnerator)

A transformer protein language diffusion model to create all-atom IDP ensembles and IDR disordered ensembles that maintain their folded domains.

## Getting started

To get started, this repository must be cloned using the following command:

```bash
git clone https://github.com/THGLab/IDPForge.git
```

Following that, the working conda environment can be established in two ways.

### IDPForge Main Installation Protocol

First, navigate to the new `IDPForge` directory:

```bash
cd IDPForge
```

The base environment can be built manually via the `environment.yml` file in the repo. To do this, run the following command:

```bash
conda env create -f environment.yml
```

> Note: The default `environment.yml` file is set to install `torch==2.5.1 and cuda==12.1` for earlier GPUs (sm_60 - sm_80). If you have newer GPUs (released after Q4 2025) and run into issues with the 12.1 installation, switch to `torch==2.7.1 and cuda==12.8`. Although the 12.8 build nominally covers sm_60 - sm_120, it is not fully backwards compatible with older architectures, so the 12.1 default is preferred for earlier GPUs. Refer to the comments in the file for modification instructions.

Once the environment is created, activate it.

```bash
conda activate IDPForge
```

Then install IDPForge as a module in the environment.

```bash
pip install -e .
```

This repo also requires `OpenFold` utilities, so that repository must be cloned in the same directory as IDPForge. To do this, first navigate to the parent directory.

```bash
cd ../
```

Then clone the OpenFold repository into the parent directory.

```bash 
git clone https://github.com/aqlaboratory/openfold.git
```

Once the repository is cloned, proceed into the resources of OpenFold.

```bash
cd openfold/openfold/resources
```

In there, download the following file.  

```bash
wget https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
```

Once this is done, navigate back into the main OpenFold directory.

```bash
cd ../../
```

The OpenFold setup must be replaced. To do this, first locate the 2 setup replacements provided with the IDPForge repository.

```bash
ls path/to/my/IDPForge/dockerfiles/openfold_setup_12*
```

> Note: The output should look like the following:
*path/to/my/IDPForge/dockerfiles/openfold_setup_12.1.py*
*path/to/my/IDPForge/dockerfiles/openfold_setup_12.8.py*

Then copy `openfold_setup_12.1.py` into the OpenFold directory as the new `setup.py`.

```bash
cp path/to/my/IDPForge/dockerfiles/openfold_setup_12.1.py path/to/my/openfold/setup.py
```

> Note: If the alternative installation was chosen during the setup of the IDPForge environment, copy the `openfold_setup_12.8.py` version instead.

Finally, install OpenFold as a module in the environment.

```bash
pip install -e .
```

This makes the environment fully ready for use.

### Alternative Installation for Compute Cluster

If you have issues setting up the base environment from the yml file, or if you are setting IDPForge up for use on an HPC cluster, it is recommended to follow the installation by openfold. To do this, start by cloning both repositories in the same directory.

```bash
git clone https://github.com/THGLab/IDPForge.git
```

```bash
git clone https://github.com/aqlaboratory/openfold.git
```

Then navigate into the OpenFold directory.

```bash
cd openfold/
``` 

First, make a copy of the `environment.yml` file without `flash-attn` so it is not installed during environment creation.

```bash
python - <<'PY'
from pathlib import Path
src = Path("environment.yml")
dst = Path("environment_noflash.yml")
lines = src.read_text().splitlines()
lines = [ln for ln in lines if "flash-attn" not in ln]
dst.write_text("\n".join(lines) + "\n")
print("Wrote", dst)
PY
```

Then create the OpenFold environment from the stripped file.

```bash
mamba env create -n openfold_env -f environment_noflash.yml
```
> Note: This can also be run with `conda env create -n openfold_env -f environment_noflash.yml`

Then activate the environment.

```bash
conda activate openfold_env
```

Install other dependencies required by IDPForge using the following commands:

```bash
conda install einops mdtraj pdb-tools -c conda-forge
``` 

```bash
conda install mmseqs2 -c bioconda
``` 

```bash
pip install tensorboard topoly
``` 

Navigate into the resources of OpenFold.

```bash
cd openfold/resources
```

In there, download the following file.  

```bash
wget https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
```

Navigate back to the main directory of OpenFold.

```bash
cd ../../
```

Install OpenFold as a module in the environment.

```bash
pip install . --no-build-isolation
```
> Note: If `pip install . --no-build-isolation` does not work, proceed with `pip install -e .` instead.

Navigate to the IDPForge directory.

```bash
cd ../IDPForge
```

Install IDPForge as a module.

```bash
pip install . --no-build-isolation
```
> Note: If `pip install . --no-build-isolation` does not work, proceed with `pip install -e .` instead.

This makes the environment fully ready for use.

> Note: For more information on OpenFold installation, please refer to the installation guide. https://openfold.readthedocs.io/en/latest/Installation.html

## Downloading model weights and other files

Model weights, example training data, and other inference input files can be downloaded from [Figshare](https://doi.org/10.6084/m9.figshare.28414937). 

It is recommended to copy the `weights/` directory directly into the IDPForge repository as `IDPForge/weights/`. Similarly, the contents of `data/` can be copied into the given `IDPForge/data/` directory.

## Notes on ESM2 and Attention

ESM2 utilities are refactored into this repo for network modules and exploring the effects of ESM embedding on IDP modeling. Alternatively, it can be installed from their GitHub https://github.com/facebookresearch/esm.git, or via pip install `pip install fair-esm`.

Optional: `pip install flash-attn==2.3` to speed up attention calculation.

## Using Docker
IDPForge can also be built as a docker container using either of the included dockerfiles (Blackwell or Ampere). Blackwell runs on CUDA12.8 and Ampere runs on CUDA12.1. Optionally, the training weights and data files from [Figshare](https://doi.org/10.6084/m9.figshare.28414937) may be merged before the creation of the image. This will ensure the image contains the merged files, removing the need for additional /weights and /data mounting.

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
conda activate IDPForge
python train.py --model_config_path configs/train.yml
```

## Sampling

Sampling loops through three phases until the target is met:

1. **Generate + Relax**: Calls `sample_ldr.py` to produce diffusion conformers, which are immediately relaxed via AMBER minimization (relax config loaded from `configs/sample.yml`).
2. **Repair**: Checks each relaxed structure for D-amino acids (chirality) and broken HIS ring bonds. Applies fixes and re-relaxes if any repairs were made.
3. **Validate**: Runs unified validation checking chirality, bond integrity, clash score (adaptive smart threshold), and backbone topology (knot detection). Passing structures are renamed to `N_validated.pdb`.

With that, sampling scripts are provided for wholly disordered and partially disordered proteins below.

### Single chain IDP/IDRs

We provide a commandline interface to sample single chain IDP/IDRs.
```
usage: sample_idp.py seq ckpt_path output_dir sample_cfg
[-h] [--batch BATCH] [--nconf NCONF] [--cuda] 
[--verbose]

positional arguments:
  seq                protein sequence
  ckpt_path          path to model weights
  output_dir         directory to output pdbs
  sample_cfg         path to a sampling configuration    
                     yaml file

optional arguments:
  --batch BATCH      batch size 
  --nconf NCONF      number of conformers to sample
  --cuda             whether to use cuda or cpu
  --verbose          show or hide debugging logs
```

Example to generate 10 conformers for Sic1:

```bash
mkdir test
sequence="GSMTPSTPPRSRGTRYLAQPSGNTSSSALMQGQKTPQKPSQNLVPVTPSTTKSFKNAPLLAPPNSNMGMTSPFNGLTSPQRSPFPKSSVKRT"
python sample_idp.py $sequence weights/mdl.ckpt test configs/sample.yml --nconf 10 --batch 4 --cuda --verbose
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
    python -u /app/sample_idp.py $sequence /app/weights/mdl.ckpt /app/output /app/configs/sample.yml --nconf 10 --batch 4 --cuda --verbose
```

### IDRs with folded domains

First, to prepare the folded template, run `python init_ldr_template.py`. We provide an example for sampling the low confidence region of AF entry P05231:
```bash
python mk_ldr_template.py data/AF-P05231-F1-model_v4.pdb 1-41 data/AF-P05231_ndr.npz
```
The provided model weights are not recommended for predicting multiple domains at the same time.

To generate an ensemble of IDRs with folded domains, run:
```bash
mkdir P05231_build
python sample_ldr.py weights/mdl.ckpt data/AF-P05231_ndr.npz P05231_build configs/sample.yml --nconf 10 --batch 4 --cuda --verbose
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
    python -u /app/sample_ldr.py /app/weights/mdl.ckpt /app/data/AF-P05231_ndr.npz /app/output /app/configs/sample.yml --nconf 10 --batch 4 --cuda --verbose
```

### Chemical shifts prediction and evaluating ensembles with X-EISD (optional)

We use UCBShift for chemical shift prediction and can be installed at https://github.com/THGLab/CSpred.git. If you wish to use X-EISD for evaluation or reweighing with experimental data, please refer to https://github.com/THGLab/X-EISDv2.

## Citation
```bibtex
@article {DeCastro2026,
	author = {De Castro, Stefano and Zhang, Oufan and Liu, Zi Hao and Forman-Kay, Julie Deborah and Head-Gordon, Teresa},
	title = {IDPForge: Deep Learning of Proteins with Global and Local Regions of Disorder},
	elocation-id = {2026.03.25.714313},
	year = {2026},
	doi = {10.64898/2026.03.25.714313},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/03/27/2026.03.25.714313},
	eprint = {https://www.biorxiv.org/content/early/2026/03/27/2026.03.25.714313.full.pdf},
	journal = {bioRxiv}
}
```
