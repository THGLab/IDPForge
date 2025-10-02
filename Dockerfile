# NOTE: RUN THIS IN "IDPForge"


# Start from CUDA 12.8 so that its guaranteed
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Add metadata for the image. 
LABEL org.opencontainers.image.authors="O Zhang"

# Install essential system tools in a single, clean layer to reduce image size.
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libxml2 \
    && rm -rf /var/lib/apt/lists/*

# Set up miniconda
ENV CONDA_DIR="/opt/conda"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# Auto-accept tos
RUN conda tos accept

# Install PyTorch using pip to get the latest CUDA 12.8-specific version
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install the rest of the dependencies using env.yml
COPY ./env.yml /opt/idpforge/env.yml
RUN conda install -n base mamba -c conda-forge && \
    conda env update -n base --file /opt/idpforge/env.yml --solver=libmamba && \
    conda clean --all

# Install/upgrade essential Python build tools before compiling OpenFold from source
RUN pip install --upgrade pip setuptools wheel

# Clone OpenFold 
RUN git clone https://github.com/aqlaboratory/openfold.git /opt/openfold

# Move modified OpenFold into the newly cloned OpenFold directory
COPY ./openfold_setup.py /opt/openfold/setup.py
RUN wget -q -P /opt/openfold/openfold/resources \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt && \
    cd /opt/openfold && \
    pip install .

# Copy the main application code and set the working directory. 
COPY . /app
WORKDIR /app