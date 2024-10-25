#!/bin/bash
set -e -x    # Exit on first error (-e) and print each command before execution (-x)

# List build machine info
lscpu
free -h

# Install lsof, setup.py need this command
yum install lsof -y

# Download and install conda
cd /root
if ! test -f ./Miniconda3-latest-Linux-x86_64.sh; then
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
fi
if ! test -d /root/miniconda3; then
    ./Miniconda3-latest-Linux-x86_64.sh -b
    export PATH=/root/miniconda3/bin:$PATH
    conda init
    source /root/.bashrc
fi

# Build package for different versions
export BUILD_HIGP_RELEASE=1
for PY_VERSION in 3.7 3.8 3.9 3.10 3.11 3.12
do
    printf "========== Building wheel for Python $PY_VERSION ==========\n"

    # Setup conda environment
    conda create -n py$PY_VERSION python=$PY_VERSION -y
    conda activate py$PY_VERSION
    conda install "blas=*=openblas" -y
    conda install numpy -y

    # Compile the python module
    cd /io/py-interface
    rm -rf build dist *.egg-info
    python setup.py bdist_wheel
    cp dist/*-linux_x86_64.whl /

    # Audit for manylinux
    cd /
    auditwheel repair *-linux_x86_64.whl
    mv /wheelhouse/*-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /io
    rm -rf *-linux_x86_64.whl

    # Done for this Python version
    conda deactivate
done
