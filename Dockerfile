FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Metadata
LABEL org.opencontainers.image.authors="O Zhang"

# Install utilities and CUDA libs
RUN apt-get update && apt-get install -y wget git libxml2 \
 && apt-key del 7fa2af80 || true \
 && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
 && dpkg -i cuda-keyring_1.0-1_all.deb \
 && apt-get update && apt-get install -y \
    cuda-minimal-build-12-1 libcusparse-dev-12-1 libcublas-dev-12-1 libcusolver-dev-12-1

# Install Miniforge
RUN wget -P /tmp \
    "https://github.com/conda-forge/miniforge/releases/download/23.3.1-1/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:$PATH"

# Install mamba and environment
COPY env.yml /opt/idpforge/env.yml
RUN conda install -n base mamba -c conda-forge
RUN mamba env update -n base --file /opt/idpforge/env.yml && conda clean --all
RUN export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# Install OpenFold
RUN git clone https://github.com/aqlaboratory/openfold.git /opt/openfold
RUN wget -q -P /opt/openfold/openfold/resources \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
WORKDIR /opt/openfold
RUN python3 setup.py install

# Copy your idpforge
COPY . /opt/idpforge
WORKDIR /opt/idpforge


