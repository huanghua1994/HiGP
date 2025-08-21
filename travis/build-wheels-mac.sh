#!/bin/bash
set -e -x    # Exit on first error (-e) and print each command before execution (-x)

# Check platform
uname -a
sw_vers

# Get repository root  
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Initialize conda (assuming conda is already in PATH)
eval "$(conda shell.bash hook)"

# Build package for different versions
export BUILD_HIGP_RELEASE=1
for PY_VERSION in 3.9 3.10 3.11 3.12
do
    printf "========== Building wheel for Python $PY_VERSION ==========\n"
    
    # Setup conda environment
    conda create -n py$PY_VERSION python=$PY_VERSION -y
    conda activate py$PY_VERSION
    
    # Install dependencies
    pip install --upgrade pip wheel setuptools
    pip install numpy torch torchvision
    
    # Compile the python module
    cd "$REPO_ROOT/py-interface"
    rm -rf build dist *.egg-info
    python setup.py bdist_wheel
    
    # Fix OpenMP dependency path
    ORIGINAL_WHEEL=$(ls dist/*.whl | head -1)
    WHEEL_NAME=$(basename "$ORIGINAL_WHEEL")
    
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    unzip -q "$REPO_ROOT/py-interface/$ORIGINAL_WHEEL"
    
    SO_FILE=$(find . -name "higp_cext*.so" | head -1)
    if [[ -n "$SO_FILE" ]]; then
        CURRENT_LIBOMP=$(otool -L "$SO_FILE" | grep libomp | awk '{print $1}' | head -1)
        if [[ -n "$CURRENT_LIBOMP" ]]; then
            install_name_tool -change "$CURRENT_LIBOMP" "@loader_path/torch/lib/libomp.dylib" "$SO_FILE"
        fi
    fi
    
    # Repackage wheel
    zip -r "$REPO_ROOT/$WHEEL_NAME" .
    cd "$REPO_ROOT"
    rm -rf "$TEMP_DIR"
    
    # Done for this Python version
    conda deactivate
done