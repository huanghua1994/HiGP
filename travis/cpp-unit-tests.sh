#!/bin/bash
set -e -x    # Exit on first error (-e) and print each command before execution (-x)

# List running environment
lscpu
free -h

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

# Download MKL using conda
conda create -n mkl -y
conda activate mkl
conda install mkl -y

# Generate makefile.in
cd /io
mv makefile.in makefile.in.bak
touch makefile.in
printf "CC  = gcc\n" >> makefile.in
printf "CXX = g++\n" >> makefile.in
printf "\n" >> makefile.in
printf "USE_MKL      = 1\n" >> makefile.in
printf "USE_OPENBLAS = 0\n" >> makefile.in
printf "\n" >> makefile.in
printf "DEFS = -DUSE_MKL\n" >> makefile.in
printf "CPU_LINALG_LD_PATH = /root/miniconda3/envs/mkl/lib\n" >> makefile.in
printf "CPU_LINALG_LIBS    = -lmkl_rt\n" >> makefile.in

# Compile C++ library and C++ test programs
cd /io/cpp-src
make -j
cd ../cpp-tests
make -j

# Recover the original makefile.in
rm -f ../makefile.in
mv ../makefile.in.bak ../makefile.in

# Run C++ tests
export LD_LIBRARY_PATH=/root/miniconda3/envs/mkl/lib:$LD_LIBRARY_PATH
./test_kernels.exe 100 100 8 bin_test_data/test_kernels_100_100_8.bin
./test_kernels.exe 100 120 3 bin_test_data/test_kernels_100_120_3.bin
./test_kernels.exe 110 103 5 bin_test_data/test_kernels_110_103_5.bin

./test_dkmat.exe 100 100 8 5 bin_test_data/test_dkmat_100_100_8_5.bin
./test_dkmat.exe 110 100 3 5 bin_test_data/test_dkmat_110_100_3_5.bin
./test_dkmat.exe 102 105 6 8 bin_test_data/test_dkmat_102_105_6_8.bin

./test_tsolvers.exe 1000 3  10 5 bin_test_data/test_tsolvers_1000_3_10_5.bin
./test_tsolvers.exe 500  2  10 8 bin_test_data/test_tsolvers_500_2_10_8.bin
./test_tsolvers.exe 2000 16 10 8 bin_test_data/test_tsolvers_2000_16_10_8.bin

./test_csr_trsm.exe 50000   500000  8  5
./test_csr_trsm.exe 1000000 5000000 10 5

./test_octree.exe 10000 2
./test_octree.exe 40000 3

./test_id_ppqr.exe 600 600 1e-4
./test_id_ppqr.exe 500 700 1e-8
./test_id_ppqr.exe 600 514 1e-10

./test_h2mat.exe 10000 2 1 2.0 1.5 1e-3 1e-4
./test_h2mat.exe 20000 3 1 2.0 1.5 1e-3 1e-9

./test_nys.exe 1000 16 8  20 bin_test_data/test_nys_1000_16_8_20.bin
./test_nys.exe 1000 3  10 40 bin_test_data/test_nys_1000_3_10_40.bin

./test_afn.exe 1000 3  5 100 50 bin_test_data/test_afn_1000_3_5_100_50.bin
./test_afn.exe 1000 12 8 50  20 bin_test_data/test_afn_1000_12_8_50_20.bin

./test_bpcg.exe 10000 3 10 1 2   1e-4
./test_bpcg.exe 5000  8 10 2 0.5 1e-4