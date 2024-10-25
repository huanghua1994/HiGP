#ifndef __HIGP_CEXT_H__
#define __HIGP_CEXT_H__

/*
 * @file higp_cext.h
 * @brief This file is the header file for the HiGP C extension module.
 * @details This file is the header file for the HiGP C extension module. \n
 *          It includes data structures for the h2 matrices and the preconditioner. \n
 *          Currently we only provide the numpy interface.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
//#include <torch/extension.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../cpp-src/common.h"
#include "../cpp-src/utils.h"
#include "../cpp-src/kernels/kernels.h"
#include "../cpp-src/dense_kernel_matrix.h"
#include "../cpp-src/h2mat/h2mat.h"
#include "../cpp-src/solvers/solvers.h"
#include "../cpp-src/gp/gp.h"

/*
 * @brief       The struct for data types.
 * @details     The struct for data types. Currently we only support FP32 and FP64.
 */
typedef enum
{
    FP32 = 0,
    FP64
} dtype_enum;

/*------------------------------------Kernel Matrix Object------------------------------------*/

/*
 * @brief       The struct for a kernel matrix.
 * @details     The struct for a kernel matrix. Can be used to store both h2 matrix and the standard dense matrix.
 */
typedef struct 
{
    PyObject_HEAD
    int _h2;
    int _nrow;
    int _ncol;
    void* _mat;
    void* _octree;
    dtype_enum _dtype;
} KrnlMatObject;

/*
 * @brief       Matrix multiplication method for a kernel matrix struct.
 * @details     Matrix multiplication method for a kernel matrix struct.
 */
static PyObject* KrnlMatObject_matmul(KrnlMatObject* self, PyObject* args, PyObject *kwds);

static char HiGP_Cext_krnlmat_matmul_help[] = 
    "Multiplies a kernel matrix K with a dense general matrix X.n"
    "Input:\n"
    "    X : Row-major NumPy array of length N or size nvecs-by-N (each row is an input vector).\n"
    "Output:\n"
    "    Y : Row-major NumPy array of K * X, same shape as the input X.\n";

/*
 * @brief       Matrix multiplication method for a kernel matrix struct with gradient.
 * @details     Matrix multiplication method for a kernel matrix struct with gradient.
 */
static PyObject* KrnlMatObject_matmul_grad(KrnlMatObject* self, PyObject* args, PyObject *kwds);

static char HiGP_Cext_krnlmat_matmul_grad_help[] = 
    "Multiplies a kernel matrix K and/or its derivates dK / d{l, f, s} with a dense general matrix X.\n"
    "Input:\n"
    "    X: Row-major NumPy array of length N or size nvecs-by-N (each row is an input vector).\n"
    "Output:\n"
    "    result: tuple of size 4\n"
    "      result[0] : K * X,     same shape as the input X.\n"
    "      result[1] : dK/dl * X, same shape as the input X.\n"
    "      result[2] : dK/df * X, same shape as the input X.\n"
    "      result[3] : dK/ds * X, same shape as the input X.\n";

/*
 * @brief       Defining methods for the kernel matrix struct.
 * @details     Defining methods for the kernel matrix struct.
 */
static PyMethodDef KrnlMatObject_methods[] = 
{
    {"matmul", (PyCFunction) KrnlMatObject_matmul, METH_VARARGS|METH_KEYWORDS, HiGP_Cext_krnlmat_matmul_help},
    {"matmul_grad", (PyCFunction) KrnlMatObject_matmul_grad, METH_VARARGS|METH_KEYWORDS, HiGP_Cext_krnlmat_matmul_grad_help},
    {NULL}
};

/*------------------------------------Kernel Matrix Module------------------------------------*/

/*
 * @brief       Initialize a kernel matrix object.
 * @details     Initialize a kernel matrix object.
 */
static int krnlmat_init(KrnlMatObject* self, PyObject* args, PyObject* kwds);

/*
 * @brief       Setup and return a kernel matrix object.
 * @details     Setup and return a kernel matrix object.
 */
static PyObject* krnlmat_setup(PyObject* self, PyObject* args, PyObject *kwds);

/*
 * @brief       Deallocate the a kernel matrix object.
 * @details     Deallocate the a kernel matrix object.
 */
static void krnlmat_dealloc(KrnlMatObject* self);

static char HiGP_Cext_krnlmat_setup_help[] = 
    "Initialize a HiGP C extension kernel matrix object.\n"
    "Inputs:\n"
    "    data           : 1st dataset point coordinate (row-major NumPy array, size dim-by-N).\n"
    "    kernel_type    : 1: Gaussian; 2: Matern 3/2; 3: Matern 5/2; 99: custom.\n"
    "    l              : The lengthscale of the kernel matrix.\n"
    "    f              : The scale of the kernel matrix.\n"
    "    s              : The noise level of the kernel matrix.\n"
    "Optional inputs (default value): \n"
    "    data2 (None)       : 2nd dataset point coordinate (row-major NumPy array, size dim-by-N). If provided, the kernel matrix is K(data, data2).\n"
    "    use_h2 (1)         : 0: Do not use H2 matrix; 1: use H2 matrix when possible.\n"
    "    nthreads (-1)      : Max number of OpenMP threads, -1 for system default number of threads.\n"
    "    h2_tol (1e-8)      : H2 matrix accuracy level.\n"
    "    h2_leaf_nmax (400) : Max number of data points in a H2 matrix leaf node. We recommend that you keep the default value.\n"
    "    h2_leaf_emax (0)   : Min enclosing box size of a H2 matrix leaf node. We recommend that you keep the default value.\n"
    "    seed (-1)          : Random number generator seed, -1 for system default number.\n"
    "Output:\n"
    "   krnlmat : A initialized matrix object.\n";

/*
 * @brief       Defining methods for the kernel matrix module.
 * @details     Defining methods for the kernel matrix module.
 */
static PyMethodDef KrnlMatMethods[] = 
{
    {"setup", (PyCFunction) krnlmat_setup, METH_VARARGS|METH_KEYWORDS, HiGP_Cext_krnlmat_setup_help},
    {NULL}
};

/*
 * @brief       The struct for the kernel matrix module.
 * @details     The struct for the kernel matrix module.
 */
static struct PyModuleDef krnlmatmodule = 
{
    PyModuleDef_HEAD_INIT,
    "krnlmat",
    NULL,
    -1,
    KrnlMatMethods
};

/*
 * @brief       Types initialization of the kernel matrix module.
 * @details     Types initialization of the kernel matrix module.
 */
static PyTypeObject KrnlMatObjectType = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
    "higp_cext.krnlmat.KrnlMatObject",
    sizeof(KrnlMatObject),
    0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "KrnlMatObject",
    .tp_methods = KrnlMatObject_methods,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) krnlmat_init,
    .tp_dealloc = (destructor) krnlmat_dealloc,
};

/*------------------------------------Preconditioner Object------------------------------------*/

/*
 * @brief       The struct for a preconditioner.
 * @details     The struct for a preconditioner. Currently only supports the AFN preconditioner.
 */
typedef struct 
{
    PyObject_HEAD
    int _precond_type;  // 0: AFN
    int _n;             // Matrix size
    void *_prec;        // Preconditioner data structure   
    dtype_enum _dtype;  // FP32 and FP64
} PrecondObject;

static char HiGP_Cext_precond_matmul_help[] = 
    "Apply preconditioner on one or multiples vectors X as M^{-1} * X.\n"
    "Inputs:\n"
    "    X : Row-major NumPy array of length N or size nvecs-by-N (each row is an input vector).\n"
    "Output:\n"
    "    Y : Row-major NumPy array of M^{-1} * X, same shape as the input X.\n";

/*
 * @brief       Apply solve with a preconditioner.
 * @details     Apply solve with a preconditioner.
 */
static PyObject* PrecondObject_matmul(PrecondObject* self, PyObject* args, PyObject *kwds);

/*
 * @brief       Defining methods for the preconditioner struct.
 * @details     Defining methods for the preconditioner struct.
 */
static PyMethodDef PrecondObject_methods[] = 
{
    {"matmul", (PyCFunction) PrecondObject_matmul, METH_VARARGS|METH_KEYWORDS, HiGP_Cext_precond_matmul_help},
    {NULL}
};

/*------------------------------------Preconditioner Module------------------------------------*/

/*
 * @brief       Initialize a preconditioner struct.
 * @details     Initialize a preconditioner struct.
 */
static int precond_init(PrecondObject* self, PyObject* args, PyObject* kwds);

/*
 * @brief       Setup and return a preconditioner struct.
 * @details     Setup and return a preconditioner struct.
 */
static PyObject* precond_setup(PyObject* self, PyObject* args, PyObject *kwds);

/*
 * @brief       Deallocate a preconditioner struct.
 * @details     Deallocate a preconditioner struct.
 */
static void precond_dealloc(PrecondObject* self);

static char HiGP_Cext_afn_setup_help[] = 
    "Setup and return a HiGP C extension AFN preconditioner struct.\n"
    "Inputs:\n"
    "    data           : Dataset point coordinate (row-major NumPy array, size dim-by-N).\n"
    "    kernel_type    : 1: Gaussian; 2: Matern 3/2; 3: Matern 5/2; 99: custom.\n"
    "    l              : The lengthscale of the kernel matrix.\n"
    "    f              : The scale of the kernel matrix.\n"
    "    s              : The noise level of the kernel matrix.\n"
    "Optional inputs (default value): \n"
    "    nthreads (-1)  : Max number of OpenMP threads, -1 for system default number of threads.\n"
    "    rank (50)      : The rank of the AFN preconditioner.\n"
    "    lfil (0)       : The fill-level of the Schur complement of the AFN preconditioner.\n"
    "    seed (-1)      : Random number generator seed, -1 for system default number.\n"
    "Output:\n"
    "   precond : An AFN preconditioner object.\n";

/*
 * @brief       Defining methods for the preconditioner struct.
 * @details     Defining methods for the preconditioner struct.
 */
static PyMethodDef PrecondMethods[] = 
{
    {"setup_afn", (PyCFunction) precond_setup, METH_VARARGS|METH_KEYWORDS, HiGP_Cext_afn_setup_help},
    {NULL}
};

/*
 * @brief       The struct for the preconditioner module.
 * @details     The struct for the preconditioner module.
 */
static struct PyModuleDef precondmodule = 
{
    PyModuleDef_HEAD_INIT,
    "precond",
    NULL,
    -1,
    PrecondMethods
};

/**
 * @brief       Types initialization of the preconditioner module.
 * @details     Types initialization of the preconditioner module.
 */
static PyTypeObject PrecondObjectType = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
    "higp_cext.precond.PrecondObject",
    sizeof(PrecondObject),
    0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "PrecondObject",
    .tp_methods = PrecondObject_methods,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) precond_init,
    .tp_dealloc = (destructor) precond_dealloc,
};

/*------------------------------------GPR Problem Object------------------------------------*/

/*
 * @brief       The struct for the GP Regression (GPR) problem.
 * @details     The struct for the GP Regression (GPR) problem.
 */
typedef struct
{
    PyObject_HEAD
    int _n;
    int _krnl;
    int _transform;
    dtype_enum _dtype;  // FP32 and FP64

    int _exact_gp;
    int _pt_dim;
    void *_X_train;     // ldim is always n_train
    void *_Y_train;     // We take NumPy array so type is same as X_train
    
    void *_params;
    void *_loss;
    void *_grad;
    int _norun;         // Track if this is the first run (if so no history is available)

    void *_pgp_loss;
} GPRProblemObject;

/*
 * @brief       Compute loss.
 * @details     Compute loss.
 */
static PyObject* GPRProblemObject_loss(GPRProblemObject* self, PyObject* args, PyObject *kwds);

static char HiGP_Cext_gpr_loss_help[] = 
    "GP Regression loss function.\n"
    "Input:\n"
    "    pyparams: The parameters (l, f, s) of the GP model (NumPy array, length 3, before transformation).\n"
    "Output:\n"
    "    loss: The loss.\n";

/*
 * @brief       Compute both loss and gradient in a tuple (loss, grad).
 * @details     Compute both loss and gradient in a tuple (loss, grad).
 */
static PyObject* GPRProblemObject_grad(GPRProblemObject* self, PyObject* args, PyObject *kwds);

static char HiGP_Cext_gpr_grad_help[] = 
    "GP regression loss function with gradient.\n"
    "Input:\n"
    "    pyparams: The parameters (l, f, s) of the GP model (NumPy array, length 3, before transformation).\n"
    "Output:\n"
    "    results: Tuple of length 2."
    "        results[0] : The loss.\n"
    "        results[1] : NumPy array of length 3, gradients of the loss.\n";

/*
 * @brief       Tell if the GPR problem is using double-precision data type.
 * @details     Tell if the GPR problem is using double-precision data type.
 */
static PyObject* GPRProblemObject_is_double(GPRProblemObject* self);

static char HiGP_Cext_gpr_is_double[] = 
    "Tell if a GP regression object is using double-precision data types.\n"
    "Output: True or false.\n";

/*
 * @brief       Get the number of training data points.
 * @details     Get the number of training data points.
 */
static PyObject* GPRProblemObject_get_n(GPRProblemObject* self);

static char HiGP_Cext_gpr_get_n[] = 
    "Get the number of training data points.\n"
    "Output: The number of training data points.\n";

/*
 * @brief       Defining methods for the GP problem struct.
 * @details     Defining methods for the GP problem struct.
 */
static PyMethodDef GPRProblemObject_methods[] = 
{
    {"loss", (PyCFunction) GPRProblemObject_loss, METH_VARARGS|METH_KEYWORDS, HiGP_Cext_gpr_loss_help},
    {"grad", (PyCFunction) GPRProblemObject_grad, METH_VARARGS|METH_KEYWORDS, HiGP_Cext_gpr_grad_help},
    {"is_double", (PyCFunction) GPRProblemObject_is_double, METH_NOARGS, HiGP_Cext_gpr_is_double},
    {"get_n", (PyCFunction) GPRProblemObject_get_n, METH_NOARGS, HiGP_Cext_gpr_get_n},
    {NULL}
};

/*------------------------------------GPR Problem Module------------------------------------*/

/*
 * @brief       Initialize the GPR problem module.
 * @details     Initialize the GPR problem module.
 */
static int gpr_problem_init(GPRProblemObject* self, PyObject* args, PyObject* kwds);

static char HiGP_Cext_gpr_problem_setup_help[] = 
    "GP regression problem struct creation function.\n"
    "Inputs:\n"
    "    data           : Dataset point coordinate (row-major NumPy array, size dim-by-N).\n"
    "    label          : The training label (NumPy array, length N).\n"
    "    kernel_type    : 1: Gaussian; 2: Matern 3/2; 3: Matern 5/2; 99: custom.\n"
    "Optional inputs (default value): \n"
    "    nthreads (-1)  : Max number of OpenMP threads, -1 for system default number of threads.\n"
    "    exact_gp (0)   : 0: use iterative methods; 1: use exact solve.\n"
    "    mvtype (0)     : Matvec type: 0: Use H2 when possible, otherwise falls back to OTF or AOT; 1: AOT (ahead-of-time); 2: OTF (on-the-fly).\n"
    "    rank (50)      : The rank of the AFN preconditioner.\n"
    "    lfil (0)       : The fill-level of the Schur complement of the AFN preconditioner.\n"
    "    niter (10)     : The number of iterations for the Lanczos Quadrature.\n"
    "    nvec (10)      : The number of test vectors for the Lanczos Quadrature.\n"
    "    seed (-1)      : Random number generator seed, -1 for system default number.\n"
    "Output:\n"
    "    prediction: tuple with two elements.\n"
    "        prediction[0] : The predicted label (NumPy array, length N).\n"
    "        prediction[1] : The standard deviation of the predicted label (NumPy array, length N).\n";

/*
 * @brief       Setup and return a GPR problem object.
 * @details     Setup and return a GPR problem object.
 */
static PyObject* gpr_problem_setup(PyObject* self, PyObject* args, PyObject *kwds);

/*
 * @brief       Deallocate the GPR problem object.
 * @details     Deallocate the GPR problem object.
 */
static void gpr_problem_dealloc(GPRProblemObject* self);

/*
 * @brief       Defining methods for the GP problem module.
 * @details     Defining methods for the GP problem module.
 */
static PyMethodDef GPRProblemMethods[] =
{
    {"setup", (PyCFunction) gpr_problem_setup, METH_VARARGS|METH_KEYWORDS, HiGP_Cext_gpr_problem_setup_help},
    {NULL}
};

/*
 * @brief       The struct for the GPR problem module.
 * @details     The struct for the GPR problem module.
 */
static struct PyModuleDef gprproblemmodule =
{
    PyModuleDef_HEAD_INIT,
    "gprproblem",
    NULL,
    -1,
    GPRProblemMethods
};

/*
 * @brief       Types initialization of the GP problem module.
 * @details     Types initialization of the GP problem module.
 */
static PyTypeObject GPRProblemObjectType =
{
    PyVarObject_HEAD_INIT(NULL, 0)
    "higp_cext.gprproblem.GPRProblemObject",
    sizeof(GPRProblemObject),
    0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "GPRProblemObject",
    .tp_methods = GPRProblemObject_methods,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) gpr_problem_init,
    .tp_dealloc = (destructor) gpr_problem_dealloc,
};

/*------------------------------------GPC Problem Object------------------------------------*/

/*
 * @brief       The struct for the GP Classification (GPC) problem.
 * @details     The struct for the GP Classification (GPC) problem.
 */
typedef struct
{
    PyObject_HEAD
    // general data slots
    int _n;
    int _krnl;
    int _transform;
    int _num_classes;
    void *_params;
    void *_loss;
    void *_grad;
    int _norun;
    dtype_enum _dtype;  // FP32 and FP64
    int _exact_gp;
    int _pt_dim;
    void *_X_train;     // ldim is always n_train
    void *_Y_train;     // We take NumPy array so type is same as X_train
    void *_pgp_loss;
} GPCProblemObject;

/*
 * @brief       Compute loss.
 * @details     Compute loss.
 */
static PyObject* GPCProblemObject_loss(GPCProblemObject* self, PyObject* args, PyObject *kwds);

static char HiGP_Cext_gpc_loss_help[] = 
    "GP classification loss function.\n"
    "Input:\n"
    "    pyparams: The parameters [l1, f1, s1, l2, f2, s2, ...] of the GP model (NumPy array, length 3 * num_classes, before transformation).\n"
    "Output:\n"
    "    loss: The loss.\n";

/*
 * @brief       Compute both loss and gradient in a tuple (loss, grad).
 * @details     Compute both loss and gradient in a tuple (loss, grad).
 */
static PyObject* GPCProblemObject_grad(GPCProblemObject* self, PyObject* args, PyObject *kwds);

static char HiGP_Cext_gpc_grad_help[] = 
    "GP classification loss function with gradients.\n"
    "Input:\n"
    "    pyparams: The parameters [l1, f1, s1, l2, f2, s2, ...] of the GP model (NumPy array, length 3 * num_classes, before transformation).\n"
    "Output:\n"
    "    results: Tuple of length 2."
    "        results[0] : The loss.\n"
    "        results[1] : NumPy array of length 3 * num_classes, the gradient of the loss.\n";

/*
 * @brief       Tell if the GPC problem is using double-precision data type.
 * @details     Tell if the GPC problem is using double-precision data type.
 */
static PyObject* GPCProblemObject_is_double(GPCProblemObject* self);

static char HiGP_Cext_gpc_is_double[] = 
    "Tell if the GPC problem is using double-precision data type.\n"
    "Output: True ot false.\n";

/*
 * @brief       Get the number of training data points.
 * @details     Get the number of training data points.
 */
static PyObject* GPCProblemObject_get_n(GPCProblemObject* self);

static char HiGP_Cext_gpc_get_n[] = 
    "Get the number of training data points.\n"
    "Output: The number of training data points.\n";

/*
 * @brief       Defining methods for the GPC problem struct.
 * @details     Defining methods for the GPC problem struct.
 */
static PyMethodDef GPCProblemObject_methods[] = 
{
    {"loss", (PyCFunction) GPCProblemObject_loss, METH_VARARGS|METH_KEYWORDS, HiGP_Cext_gpc_loss_help},
    {"grad", (PyCFunction) GPCProblemObject_grad, METH_VARARGS|METH_KEYWORDS, HiGP_Cext_gpc_grad_help},
    {"is_double", (PyCFunction) GPCProblemObject_is_double, METH_NOARGS, HiGP_Cext_gpc_is_double},
    {"get_n", (PyCFunction) GPCProblemObject_get_n, METH_NOARGS, HiGP_Cext_gpc_get_n},
    {NULL}
};

/*------------------------------------GPC Problem Module------------------------------------*/

/*
 * @brief       Initialize the GPC problem module.
 * @details     Initialize the GPC problem module.
 */
static int gpc_problem_init(GPCProblemObject* self, PyObject* args, PyObject* kwds);

static char HiGP_Cext_gpc_problem_setup_help[] = 
    "GP classification problem struct creation function.\n"
    "Inputs:\n"
    "    data           : Dataset point coordinate (row-major NumPy array, size dim-by-N).\n"
    "    label          : The training label (NumPy array, length N).\n"
    "    kernel_type    : 1: Gaussian; 2: Matern 3/2; 3: Matern 5/2; 99: custom.\n"
    "Optional inputs (default value): \n"
    "    nthreads (-1)  : Max number of OpenMP threads, -1 for system default number of threads.\n"
    "    exact_gp (0)   : 0: use iterative methods; 1: use exact solve.\n"
    "    mvtype (0)     : Matvec type: 0: Use H2 when possible, otherwise falls back to OTF or AOT; 1: AOT (ahead-of-time); 2: OTF (on-the-fly).\n"
    "    rank (50)      : The rank of the AFN preconditioner.\n"
    "    lfil (0)       : The fill-level of the Schur complement of the AFN preconditioner.\n"
    "    niter (10)     : The number of iterations for the Lanczos Quadrature.\n"
    "    nvec (10)      : The number of test vectors for the Lanczos Quadrature.\n"
    "    seed (-1)      : Random number generator seed, -1 for system default number.\n"
    "Output:\n"
    "    prediction: tuple with two elements.\n"
    "        prediction[0] : The predicted label (NumPy array, length N).\n"
    "        prediction[1] : The standard deviation of the predicted label (NumPy array, length N).\n";

/*
 * @brief       Setup and return a GPC problem object.
 * @details     Setup and return a GPC problem object.
 */
static PyObject* gpc_problem_setup(PyObject* self, PyObject* args, PyObject *kwds);

/*
 * @brief       Deallocate the GPC problem object.
 * @details     Deallocate the GPC problem object.
 */
static void gpc_problem_dealloc(GPCProblemObject* self);

/*
 * @brief       Defining methods for the GPC problem module.
 * @details     Defining methods for the GPC problem module.
 */
static PyMethodDef GPCProblemMethods[] =
{
    {"setup", (PyCFunction) gpc_problem_setup, METH_VARARGS|METH_KEYWORDS, HiGP_Cext_gpc_problem_setup_help},
    {NULL}
};

/*
 * @brief       The struct for the GPC problem module.
 * @details     The struct for the GPC problem module.
 */
static struct PyModuleDef gpcproblemmodule =
{
    PyModuleDef_HEAD_INIT,
    "gpcproblem",
    NULL,
    -1,
    GPCProblemMethods
};

/*
 * @brief       Types initialization of the GPC problem module.
 * @details     Types initialization of the GPC problem module.
 */
static PyTypeObject GPCProblemObjectType =
{
    PyVarObject_HEAD_INIT(NULL, 0)
    "higp_cext.gpcproblem.GPCProblemObject",
    sizeof(GPCProblemObject),
    0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "GPCProblemObject",
    .tp_methods = GPCProblemObject_methods,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) gpc_problem_init,
    .tp_dealloc = (destructor) gpc_problem_dealloc,
};

/*------------------------------------HiGP Module------------------------------------*/

// Compute prediction
static PyObject* HiGP_Cext_gpr_prediction(PyObject* self, PyObject *args, PyObject *kwds);

static char HiGP_Cext_gpr_prediction_help[] = 
    "GP regression prediction function.\n"
    "Inputs:\n"
    "    data_train         : The training data (row-major NumPy array, size d-by-N).\n"
    "    label_train        : The training label (NumPy array, length N).\n"
    "    data_prediction    : The data used in prediction (NumPy array, size d-by-N).\n"
    "    kernel_type        : 1: Gaussian; 2: Matern 3/2; 3: Matern 5/2; 99: custom.\n"
    "    pyparams           : The parameters (l, f, s) of the GP model (NumPy array, length 3, before transformation).\n"
    "Optional inputs (default value): \n"
    "    nthreads (-1)  : Max number of OpenMP threads, -1 for system default number of threads.\n"
    "    exact_gp (0)   : 0: use iterative methods; 1: use exact solve.\n"
    "    mvtype (0)     : Matvec type: 0: Use H2 when possible, otherwise falls back to OTF or AOT; 1: AOT (ahead-of-time); 2: OTF (on-the-fly).\n"
    "    rank (50)      : The rank of the AFN preconditioner.\n"
    "    lfil (0)       : The fill-level of the Schur complement of the AFN preconditioner.\n"
    "    niter (50)     : The number of iterations for PCG in the precondition.\n"
    "    tol (1e-6)     : The tolerance for PCG in the precondition.\n"
    "Output:\n"
    "    prediction: tuple with two elements.\n"
    "        prediction[0] : The predicted label (NumPy array, length N).\n"
    "        prediction[1] : The standard deviation of the predicted label (NumPy array, length N).\n";

// Compute prediction
static PyObject* HiGP_Cext_gpc_prediction(PyObject* self, PyObject *args, PyObject *kwds);

static char HiGP_Cext_gpc_prediction_help[] = 
    "GP classification prediction function.\n"
    "Inputs:\n"
    "    data_train         : The training data (row-major NumPy array, size d-by-N).\n"
    "    label_train        : The training label (NumPy array, length N).\n"
    "    data_prediction    : The data used in prediction (NumPy array, size d-by-N).\n"
    "    kernel_type        : 1: Gaussian; 2: Matern 3/2; 3: Matern 5/2; 99: custom.\n"
    "    pyparams           : The parameters [l1, f1, s1, l2, f2, s2, ...] of the GP model (NumPy array, length 3 * num_classes, before transformation).\n"
    "Optional inputs (default value): \n"
    "    nthreads (-1)  : Max number of OpenMP threads, -1 for system default number of threads.\n"
    "    exact_gp (0)   : 0: use iterative methods; 1: use exact solve.\n"
    "    mvtype (0)     : Matvec type: 0: Use H2 when possible, otherwise falls back to OTF or AOT; 1: AOT (ahead-of-time); 2: OTF (on-the-fly).\n"
    "    nsamples (256) : Number of sample vectors for predicting probability.\n"
    "    rank (50)      : The rank of the AFN preconditioner.\n"
    "    lfil (0)       : The fill-level of the Schur complement of the AFN preconditioner.\n"
    "    niter (50)     : The number of iterations for PCG in the precondition.\n"
    "    tol (1e-6)     : The tolerance for PCG in the precondition.\n"
    "Output:\n"
    "   prediction: tuple with three elements.\n"
    "       prediction[0] : The predicted label (NumPy array, length N).\n"
    "       prediction[1] : The predicted value (row-major NumPy array, size num_classes-by-N).\n"
    "       prediction[2] : The predicted probability (row-major NumPy array, size num_classes-by-N).\n";

/*
 * @brief       Function for the global HiGP.
 * @details     Function for the global HiGP.
 */
static PyMethodDef HiGP_Cext_module_methods[] = 
{
    {"gpr_prediction", (PyCFunction) HiGP_Cext_gpr_prediction, METH_VARARGS|METH_KEYWORDS, HiGP_Cext_gpr_prediction_help},
    {"gpc_prediction", (PyCFunction) HiGP_Cext_gpc_prediction, METH_VARARGS|METH_KEYWORDS, HiGP_Cext_gpc_prediction_help},
    {NULL}
};

static PyModuleDef higp_cext_module = 
{
    PyModuleDef_HEAD_INIT,
    "higp_cext",                /* m_name */
    NULL,                       /* m_doc */
    -1,                         /* m_size */
    HiGP_Cext_module_methods,   /* m_methods */
};

/*
 * @brief     Initialize the HiGP C extension module.
 * @details   Initialize the HiGP C extension module.
 */
PyMODINIT_FUNC PyInit_higp_cext(void);

#endif // HIGP_CEXT_H
