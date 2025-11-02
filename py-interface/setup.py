import os
import platform
import numpy
import setuptools
import setuptools.command.build_py
from datetime import datetime

workdir = os.path.abspath(os.path.dirname(__file__))

# Basic C/C++ flags and linking flags
cflags  = ["-g", "-std=c++14", "-O2", "-fopenmp"]
cflags += ["-Wno-unused-result", "-Wno-unused-function", "-Wno-unused-variable", "-Wno-cpp"]
lflags  = ["-lgfortran", "-lm", '-fopenmp']

if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    # Apple Silicon: OpenMP + Accelerate dual integration
    print("Building for Apple Silicon with OpenMP + Accelerate")
    
    os.environ['MACOSX_DEPLOYMENT_TARGET'] = '13.3'
    
    # Remove default -fopenmp from cflags and lflags and add macOS-specific version
    cflags = [flag for flag in cflags if flag != "-fopenmp"]
    lflags = [flag for flag in lflags if flag != "-fopenmp"]
    cflags = [flag for flag in cflags if flag != "-std=c++14"]
    cflags += ["-Xpreprocessor", "-fopenmp"]
    
    omp_found = False
    
    # PyTorch's bundled OpenMP
    try:
        import torch
        torch_path = os.path.dirname(torch.__file__)
        torch_omp_include = os.path.join(torch_path, 'include')
        torch_omp_lib = os.path.join(torch_path, 'lib')
        torch_omp_header = os.path.join(torch_omp_include, 'omp.h')
        torch_omp_dylib = os.path.join(torch_omp_lib, 'libomp.dylib')
        
        if os.path.exists(torch_omp_header) and os.path.exists(torch_omp_dylib):
            print(f"Found OpenMP in PyTorch: {torch_omp_lib}")
            cflags += [f"-I{torch_omp_include}"]
            lflags += [f"-L{torch_omp_lib}", "-lomp"]
            omp_found = True
    except ImportError:
        pass
    
    # Fallback to Homebrew OpenMP
    if not omp_found:
        homebrew_omp_paths = [
            "/opt/homebrew/opt/libomp",  # ARM64 Homebrew
            "/usr/local/opt/libomp"      # Intel Homebrew
        ]
        
        for homebrew_path in homebrew_omp_paths:
            if os.path.exists(os.path.join(homebrew_path, "include", "omp.h")):
                print(f"Found OpenMP in Homebrew: {homebrew_path}")
                cflags += [f"-I{homebrew_path}/include"]
                lflags += [f"-L{homebrew_path}/lib", "-lomp"]
                omp_found = True
                break
        
        if not omp_found:
            print("WARNING: OpenMP not found! Please install with: brew install libomp")
            print("         Or ensure PyTorch is installed with OpenMP support")
    
    cflags += ["-DACCELERATE_NEW_LAPACK", "-DUSE_ACCELERATE_LP64"]
    lflags += ["-framework", "Accelerate"]
    
    lflags = [flag for flag in lflags if flag != "-lgfortran"]
    
    # TODO
    if "BUILD_HIGP_RELEASE" in os.environ:
        pass
    else:
        pass
    
    # Skip lsof-based BLAS detection for macOS
    skip_blas_detection = True
else:
    # For the release build, target the Haswell architecture (AVX2)
    if "BUILD_HIGP_RELEASE" in os.environ:
        cflags += ["-march=haswell"]
    else:
        cflags += ["-march=native"]
    
    skip_blas_detection = False

# Get the linalg library (MKL or OpenBLAS) numpy is using
def get_numpy_linalg_lib():
    pid = os.getpid()
    os.system('lsof -p ' + str(pid) + ' | grep -E "libmkl_rt|libopenblas" > /tmp/numpy_load.txt')
    file1 = open("/tmp/numpy_load.txt", "r")
    lsof_line = file1.readline()
    file1.close()
    os.system("rm /tmp/numpy_load.txt")
    has_mkl = 0
    has_openblas = 0
    if "libmkl_rt" in lsof_line:
        # MKL always has LP64 interface
        has_mkl = 32
    if "libopenblas64" not in lsof_line and "libopenblas" in lsof_line:
        # OpenBLAS uses LP64 (32-bit integer interface)
        has_openblas = 32
    if "libopenblas64" in lsof_line:
        # OpenBLAS uses ILP64 (64-bit integer interface)
        has_openblas = 64
    for s1 in lsof_line.split(" "):
        substr = s1.strip()
        if os.path.isfile(substr):
            lib_path = substr
            break
    lib_dir = os.path.dirname(lib_path)
    base_name = os.path.basename(lib_path)
    lib_name_noext = base_name.split(".so")[0]
    lib_name = lib_name_noext[3:]
    return has_mkl, has_openblas, lib_dir, lib_name

if not skip_blas_detection:
    has_mkl, has_openblas, lib_dir, lib_name = get_numpy_linalg_lib()
    lflags += ["-L", lib_dir, "-l", lib_name]
    if has_mkl > 0:
        cflags += ["-DUSE_MKL"]
    if has_openblas == 32:
        cflags += ["-DUSE_OPENBLAS_LP64"]
    if has_openblas == 64:
        cflags += ["-DUSE_OPENBLAS_ILP64"]
else:
    print("Skipping BLAS detection - using Accelerate Framework")

higp_cext = setuptools.Extension(
    'higp_cext',
    sources = [
        workdir + "/../cpp-src/gp/exact_gp.cpp",
        workdir + "/../cpp-src/gp/gpc_common.cpp",
        workdir + "/../cpp-src/gp/nonneg_transform.cpp",
        workdir + "/../cpp-src/gp/precond_gp.cpp",
        workdir + "/../cpp-src/kernels/custom_kernel.cpp",
        workdir + "/../cpp-src/kernels/gaussian_kernel.cpp",
        workdir + "/../cpp-src/kernels/kernels.cpp",
        workdir + "/../cpp-src/kernels/matern32_kernel.cpp",
        workdir + "/../cpp-src/kernels/matern52_kernel.cpp",
        workdir + "/../cpp-src/kernels/pdist2_kernel.cpp",
        workdir + "/../cpp-src/h2mat/h2mat_build.cpp",
        workdir + "/../cpp-src/h2mat/h2mat_matmul.cpp",
        workdir + "/../cpp-src/h2mat/h2mat_proxy_points.cpp",
        workdir + "/../cpp-src/h2mat/h2mat_typedef.cpp",
        workdir + "/../cpp-src/h2mat/h2mat_utils.cpp",
        workdir + "/../cpp-src/h2mat/id_ppqr.cpp",
        workdir + "/../cpp-src/h2mat/octree.cpp",
        workdir + "/../cpp-src/h2mat/ss_h2mat.cpp",
        workdir + "/../cpp-src/solvers/afn_precond.cpp",
        workdir + "/../cpp-src/solvers/bpcg.cpp",
        workdir + "/../cpp-src/solvers/csr_mat.cpp",
        workdir + "/../cpp-src/solvers/fsai_precond.cpp",
        workdir + "/../cpp-src/solvers/mfom.cpp",
        workdir + "/../cpp-src/solvers/mpcg.cpp",
        workdir + "/../cpp-src/solvers/nys_precond.cpp",
        workdir + "/../cpp-src/dense_kernel_matrix.cpp",
        workdir + "/../cpp-src/utils.c",
        workdir + "/higp_cext.c"
    ],
    include_dirs = [numpy.get_include()],
    extra_compile_args = cflags,
    extra_link_args = lflags,
    language = "c++"
)

setuptools.setup(
    name = "higp",
    version = datetime.today().strftime("%Y.%m.%d"),
    description = "HiGP is a high-performance Python package for using Gaussian processes (GPs) with large datasets. Its functionality includes estimating GP hyperparameters, GP regression, and GP classification. Please visit https://github.com/huanghua1994/HiGP for the source code, documentation, and examples.",
    packages = setuptools.find_packages(),
    install_requires = ["numpy>=1.24.4", "scipy>=1.3", "torch>=1.3"],
    ext_modules = [higp_cext]
)