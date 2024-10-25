#ifndef __CPU_LINALG_HPP__
#define __CPU_LINALG_HPP__

#if defined(USE_MKL) || defined(USE_OPENBLAS_LP64)
#include "cpu_linalg_lp64.hpp"
#endif

#if defined(USE_OPENBLAS_ILP64)
#include "cpu_linalg_ilp64.hpp"
#endif

#endif 