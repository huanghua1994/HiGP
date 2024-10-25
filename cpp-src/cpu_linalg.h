#ifndef __CPU_LINALG_H__
#define __CPU_LINALG_H__

#if defined(USE_MKL) || defined(USE_OPENBLAS_LP64)
#include "cpu_linalg_lp64.h"
#endif

#if defined(USE_OPENBLAS_ILP64)
#include "cpu_linalg_ilp64.h"
#endif

#endif 