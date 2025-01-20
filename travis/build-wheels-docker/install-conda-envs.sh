#!/bin/bash
set -e -x

# Download and install conda
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/Miniconda3-latest-Linux-x86_64.sh
chmod +x /tmp/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b

export PATH=/root/miniconda3/bin:$PATH
conda init
source /root/.bashrc

# Switch to conda-forge channel for libopenblas=*=*openmp*
conda config --add channels conda-forge
conda config --set channel_priority strict

for PY_VERSION in 3.8 3.9 3.10 3.11 3.12
do
    printf "========== Setting up build environment for Python $PY_VERSION ==========\n"
    conda create -n py$PY_VERSION python=$PY_VERSION -y
    conda activate py$PY_VERSION
    conda install -y libopenblas=*=*openmp*
    conda install -y "blas=*=openblas"
    conda install -y numpy
    conda deactivate
done
