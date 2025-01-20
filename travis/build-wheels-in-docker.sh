#!/bin/bash
set -e -x    # Exit on first error (-e) and print each command before execution (-x)

# List build machine info
lscpu
free -h

source /root/.bashrc

# Build package for different versions
export BUILD_HIGP_RELEASE=1
for PY_VERSION in 3.8 3.9 3.10 3.11 3.12
do
    printf "========== Building wheel for Python $PY_VERSION ==========\n"

    # Activate conda environment
    conda activate py$PY_VERSION

    # Compile the python module
    cd /io/py-interface
    rm -rf build dist *.egg-info
    python setup.py bdist_wheel
    cp dist/*-linux_x86_64.whl /

    # Audit for manylinux
    # auditwheel might be unable to find the libopenblas.so.0, add it to LD_LIBRARY_PATH
    cd /
    export LD_LIBRARY_PATH=/root/miniconda3/envs/py$PY_VERSION/lib:$OLD_LD_LIBRARY_PATH
    auditwheel repair *-linux_x86_64.whl
    mv /wheelhouse/*-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /io
    rm -rf *-linux_x86_64.whl

    # Done for this Python version
    conda deactivate
done
