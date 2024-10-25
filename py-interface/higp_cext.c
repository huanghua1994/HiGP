#include "higp_cext.h"
#include "omp.h"

/*------------------------------------ H2 Matrix Object ------------------------------------*/

static int _h2_min_size = 20000;

static PyObject* KrnlMatObject_matmul(KrnlMatObject* self, PyObject* args, PyObject *kwds)
{
    // Parse parameters
    PyArrayObject *x_vec = NULL;
    PyArrayObject *y_vec = NULL;
    static char *kwlist[] = {"x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
            &PyArray_Type, &x_vec))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }

    // Detect data type to choose from different functions
    int m = self->_nrow;
    int n = self->_ncol;

    // Check if either row/col is 1, in this case the input is a 1D array
    int xdim = PyArray_NDIM(x_vec);

    if (xdim < 1 || xdim > 2)
    {
        PyErr_SetString(PyExc_ValueError, "Support only 1D or 2D array!");
        return NULL;
    }

    int nvec = 1;
    int y_case = 0;
    if (xdim == 1)
    {
        if (PyArray_DIM(x_vec, 0) != n)
        {
            PyErr_SetString(PyExc_ValueError, "Dimension does not match!");
            return NULL;
        }
    }
    else
    {
        int dimx_m = PyArray_DIM(x_vec, 0);
        int dimx_n = PyArray_DIM(x_vec, 1);
        if (dimx_m == 1)
        {
            // This is a 1D vector, check if dimension matches
            if (dimx_n != n)
            {
                PyErr_SetString(PyExc_ValueError, "Dimension does not match!");
                return NULL;
            }
            y_case = 1;
        }
        else if (dimx_n == 1)
        {
            // This is a 1D vector, check if dimension matches
            if (dimx_m != n)
            {
                PyErr_SetString(PyExc_ValueError, "Dimension does not match!");
                return NULL;
            }
            y_case = 2;
        }
        else
        {
            // In this case, we assume each column is a vector
            nvec = PyArray_DIM(x_vec, 0);
            if (n != PyArray_DIM(x_vec, 1))
            {
                PyErr_SetString(PyExc_ValueError, "Dimension does not match!");
                return NULL;
            }
        }
    }

    // Determine the data type
    dtype_enum dtype = (PyArray_TYPE(x_vec) == NPY_FLOAT32) ? FP32 : FP64;
    if (dtype != self->_dtype)
    {
        PyErr_SetString(PyExc_ValueError, "Data type does not match!");
        return NULL;
    }

    // Determine the return type
    switch (dtype)
    {
        case FP64:
        {
            if (nvec == 1)
            {
                switch (y_case)
                {
                    case 0:
                    {
                        npy_intp dim[] = {m};
                        y_vec = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT64);
                        break;
                    }
                    case 1:
                    {
                        npy_intp dim[] = {1, m};
                        y_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                        break;
                    }
                    case 2:
                    {
                        npy_intp dim[] = {m, 1};
                        y_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                        break;
                    }
                }
            }
            else
            {
                npy_intp dim[] = {m, nvec};
                y_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
            }
            break;
        }
        case FP32:
        {
            if (nvec == 1)
            {
                switch (y_case)
                {
                    case 0:
                    {
                        npy_intp dim[] = {m};
                        y_vec = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT32);
                        break;
                    }
                    case 1:
                    {
                        npy_intp dim[] = {1, m};
                        y_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                        break;
                    }
                    case 2:
                    {
                        npy_intp dim[] = {m, 1};
                        y_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                        break;
                    }
                }
            }
            else
            {
                npy_intp dim[] = {m, nvec};
                y_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
            }
            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown data type!");
            return NULL;
        }
    }

    // double param[4] = {(double) dim, l, f, s};
    if (self->_h2)
    {
        switch (self->_dtype)
        {
            case FP64:
            {
                double *x_data = (double *) PyArray_DATA(x_vec);
                double *y_data = (double *) PyArray_DATA(y_vec);
                ss_h2mat_krnl_matmul((ss_h2mat_p) self->_mat, nvec, x_data, n, y_data, m);
                break;
            }
            case FP32:
            {
                float *x_data = (float *) PyArray_DATA(x_vec);
                float *y_data = (float *) PyArray_DATA(y_vec);
                ss_h2mat_krnl_matmul((ss_h2mat_p) self->_mat, nvec, x_data, n, y_data, m);
                break;
            }
            default:
            {
                PyErr_SetString(PyExc_ValueError, "Unknown data type!");
                return NULL;
            }
        }
    }
    else
    {
        switch (self->_dtype)
        {
            case FP64:
            {
                double *x_data = (double *) PyArray_DATA(x_vec);
                double *y_data = (double *) PyArray_DATA(y_vec);
                dense_krnl_mat_krnl_matmul((dense_krnl_mat_p) self->_mat, nvec, x_data, n, y_data, m);
                break;
            }
            case FP32:
            {
                float *x_data = (float *) PyArray_DATA(x_vec);
                float *y_data = (float *) PyArray_DATA(y_vec);
                dense_krnl_mat_krnl_matmul((dense_krnl_mat_p) self->_mat, nvec, x_data, n, y_data, m);
                break;
            }
            default:
            {
                PyErr_SetString(PyExc_ValueError, "Unknown data type!");
                return NULL;
            }
        }
    }
    
    return PyArray_Return(y_vec);
}

static PyObject* KrnlMatObject_matmul_grad(KrnlMatObject* self, PyObject* args, PyObject *kwds)
{
    // Parse parameters
    PyArrayObject *x_vec = NULL;
    PyArrayObject *y_vec = NULL, *dyl_vec = NULL, *dyf_vec = NULL, *dys_vec = NULL;
    static char *kwlist[] = {"x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyArray_Type, &x_vec))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }

    int m = self->_nrow;
    int n = self->_ncol;

    // Check x dimension
    int xdim = PyArray_NDIM(x_vec);
    if (xdim < 1 || xdim > 2)
    {
        PyErr_SetString(PyExc_ValueError, "Support only 1D or 2D array!");
        return NULL;
    }

    int nvec = 1;
    int y_case = 0;
    if (xdim == 1)
    {
        if (PyArray_DIM(x_vec, 0) != n)
        {
            PyErr_SetString(PyExc_ValueError, "Dimension does not match!");
            return NULL;
        }
    }
    else
    {
        int dimx_m = PyArray_DIM(x_vec, 0);
        int dimx_n = PyArray_DIM(x_vec, 1);
        if (dimx_m == 1)
        {
            // This is a 1D vector, check if dimension matches
            if (dimx_n != n)
            {
                PyErr_SetString(PyExc_ValueError, "Dimension does not match!");
                return NULL;
            }
            y_case = 1;
        }
        else if (dimx_n == 1)
        {
            // This is a 1D vector, check if dimension matches
            if (dimx_m != n)
            {
                PyErr_SetString(PyExc_ValueError, "Dimension does not match!");
                return NULL;
            }
            y_case = 2;
        }
        else
        {
            // In this case, we assume each column is a vector
            nvec = PyArray_DIM(x_vec, 0);
            if (n != PyArray_DIM(x_vec, 1))
            {
                PyErr_SetString(PyExc_ValueError, "Dimension does not match!");
                return NULL;
            }
        }
    }

    // Determine the data type
    dtype_enum dtype = (PyArray_TYPE(x_vec) == NPY_FLOAT32) ? FP32 : FP64;
    if (dtype != self->_dtype)
    {
        PyErr_SetString(PyExc_ValueError, "Data type does not match!");
        return NULL;
    }

    // Determine the return type
    switch (dtype)
    {
        case FP64:
        {
            if (nvec == 1)
            {
                switch (y_case)
                {
                    case 0:
                    {
                        npy_intp dim[] = {m};
                        y_vec   = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT64);
                        dyl_vec = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT64);
                        dyf_vec = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT64);
                        dys_vec = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT64);
                        break;
                    }
                    case 1:
                    {
                        npy_intp dim[] = {1, m};
                        y_vec   = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                        dyl_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                        dyf_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                        dys_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                        break;
                    }
                    case 2:
                    {
                        npy_intp dim[] = {m, 1};
                        y_vec   = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                        dyl_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                        dyf_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                        dys_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                        break;
                    }
                }
            }
            else
            {
                npy_intp dim[] = {m, nvec};
                y_vec   = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                dyl_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                dyf_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                dys_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
            }
            break;
        }
        case FP32:
        {
            if (nvec == 1)
            {
                switch (y_case)
                {
                    case 0:
                    {
                        npy_intp dim[] = {m};
                        y_vec   = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT32);
                        dyl_vec = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT32);
                        dyf_vec = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT32);
                        dys_vec = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT32);
                        break;
                    }
                    case 1:
                    {
                        npy_intp dim[] = {1, m};
                        y_vec   = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                        dyl_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                        dyf_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                        dys_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                        break;
                    }
                    case 2:
                    {
                        npy_intp dim[] = {m, 1};
                        y_vec   = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                        dyl_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                        dyf_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                        dys_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                        break;
                    }
                }
            }
            else
            {
                npy_intp dim[] = {m, nvec};
                y_vec   = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                dyl_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                dyf_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                dys_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
            }
            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown data type!");
            return NULL;
        }
    }

    // double param[4] = {(double) dim, l, f, s};
    if (self->_h2)
    {
        switch (self->_dtype)
        {
            case FP64:
            {
                double *x_data   = (double *) PyArray_DATA(x_vec);
                double *y_data   = (double *) PyArray_DATA(y_vec);
                double *dyl_data = (double *) PyArray_DATA(dyl_vec);
                double *dyf_data = (double *) PyArray_DATA(dyf_vec);
                double *dys_data = (double *) PyArray_DATA(dys_vec);
                ss_h2mat_grad_matmul((ss_h2mat_p) self->_mat, nvec, x_data, n, y_data, dyl_data, dyf_data, dys_data, n);
                break;
            }
            case FP32:
            {
                float *x_data   = (float *) PyArray_DATA(x_vec);
                float *y_data   = (float *) PyArray_DATA(y_vec);
                float *dyl_data = (float *) PyArray_DATA(dyl_vec);
                float *dyf_data = (float *) PyArray_DATA(dyf_vec);
                float *dys_data = (float *) PyArray_DATA(dys_vec);
                ss_h2mat_grad_matmul((ss_h2mat_p) self->_mat, nvec, x_data, n, y_data, dyl_data, dyf_data, dys_data, n);
                break;
            }
            default:
            {
                PyErr_SetString(PyExc_ValueError, "Unknown data type!");
                return NULL;
            }
        }
    }
    else
    {
        printf("Matvec Grad dtype %d\n",self->_dtype);
        switch (self->_dtype)
        {
            case FP64:
            {
                double *x_data   = (double *) PyArray_DATA(x_vec);
                double *y_data   = (double *) PyArray_DATA(y_vec);
                double *dyl_data = (double *) PyArray_DATA(dyl_vec);
                double *dyf_data = (double *) PyArray_DATA(dyf_vec);
                double *dys_data = (double *) PyArray_DATA(dys_vec);
                dense_krnl_mat_grad_matmul((dense_krnl_mat_p) self->_mat, nvec, x_data, n, y_data, dyl_data, dyf_data, dys_data, m);
                break;
            }
            case FP32:
            {
                float *x_data   = (float *) PyArray_DATA(x_vec);
                float *y_data   = (float *) PyArray_DATA(y_vec);
                float *dyl_data = (float *) PyArray_DATA(dyl_vec);
                float *dyf_data = (float *) PyArray_DATA(dyf_vec);
                float *dys_data = (float *) PyArray_DATA(dys_vec);
                dense_krnl_mat_grad_matmul((dense_krnl_mat_p) self->_mat, nvec, x_data, n, y_data, dyl_data, dyf_data, dys_data, m);
                break;
            }
            default:
            {
                PyErr_SetString(PyExc_ValueError, "Unknown data type!");
                return NULL;
            }
        }
    }

    PyObject* results = PyTuple_Pack(4, y_vec, dyl_vec, dyf_vec, dys_vec);
    return results;
}

/*------------------------------------ Preconditioner Object ------------------------------------*/

static PyObject* PrecondObject_matmul(PrecondObject* self, PyObject* args, PyObject *kwds)
{
    // Parse parameters
    PyArrayObject *x_vec = NULL;
    PyArrayObject *y_vec = NULL;
    static char *kwlist[] = {"x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyArray_Type, &x_vec))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }

    int n = self->_n;

    // Check x dimension
    int xdim = PyArray_NDIM(x_vec);
    if (xdim < 1 || xdim > 2)
    {
        PyErr_SetString(PyExc_ValueError, "Support only 1D or 2D array!");
        return NULL;
    }

    int nvec = 1;
    int y_case = 0;
    if (xdim == 1)
    {
        if (PyArray_DIM(x_vec, 0) != n)
        {
            PyErr_SetString(PyExc_ValueError, "Dimension does not match!");
            return NULL;
        }
    }
    else
    {
        int dimx_m = PyArray_DIM(x_vec, 0);
        int dimx_n = PyArray_DIM(x_vec, 1);
        if (dimx_m == 1)
        {
            // This is a 1D vector, check if dimension matches
            if (dimx_n != n)
            {
                PyErr_SetString(PyExc_ValueError, "Dimension does not match!");
                return NULL;
            }
            y_case = 1;
        }
        else if (dimx_n == 1)
        {
            // This is a 1D vector, check if dimension matches
            if (dimx_m != n)
            {
                PyErr_SetString(PyExc_ValueError, "Dimension does not match!");
                return NULL;
            }
            y_case = 2;
        }
        else
        {
            // In this case, we assume each column is a vector
            nvec = PyArray_DIM(x_vec, 0);
            if (n != PyArray_DIM(x_vec, 1))
            {
                PyErr_SetString(PyExc_ValueError, "Dimension does not match!");
                return NULL;
            }
        }
    }
    
    // Determine the data type
    dtype_enum dtype = (PyArray_TYPE(x_vec) == NPY_FLOAT32) ? FP32 : FP64;
    if (dtype != self->_dtype)
    {
        PyErr_SetString(PyExc_ValueError, "Data type does not match!");
        return NULL;
    }

    // Determine the return type
    switch (dtype)
    {
        case FP64:
        {
            if (nvec == 1)
            {
                switch (y_case)
                {
                    case 0:
                    {
                        npy_intp dim[] = {n};
                        y_vec = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT64);
                        break;
                    }
                    case 1:
                    {
                        npy_intp dim[] = {1,n};
                        y_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                        break;
                    }
                    case 2:
                    {
                        npy_intp dim[] = {n,1};
                        y_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
                        break;
                    }
                }
            }
            else
            {
                npy_intp dim[] = {n, nvec};
                y_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT64);
            }
            break;
        }
        case FP32:
        {
            if (nvec == 1)
            {
                switch (y_case)
                {
                    case 0:
                    {
                        npy_intp dim[] = {n};
                        y_vec = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT32);
                        break;
                    }
                    case 1:
                    {
                        npy_intp dim[] = {1,n};
                        y_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                        break;
                    }
                    case 2:
                    {
                        npy_intp dim[] = {n,1};
                        y_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
                        break;
                    }
                }
            }
            else
            {
                npy_intp dim[] = {n, nvec};
                y_vec = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
            }
            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown data type!");
            return NULL;
        }
    }

    // double param[4] = {(double) dim, l, f, s};
    switch (self->_precond_type)
    {
        case 0:
        {
            switch (self->_dtype)
            {
                case FP64:
                {
                    double *x_data = (double *) PyArray_DATA(x_vec);
                    double *y_data = (double *) PyArray_DATA(y_vec);
                    afn_precond_apply((afn_precond_p)self->_prec, nvec, x_data, n, y_data, n);
                    break;
                }
                case FP32:
                {
                    float *x_data = (float *) PyArray_DATA(x_vec);
                    float *y_data = (float *) PyArray_DATA(y_vec);
                    afn_precond_apply((afn_precond_p)self->_prec, nvec, x_data, n, y_data, n);
                    break;
                }
                default:
                {
                    PyErr_SetString(PyExc_ValueError, "Unknown data type!");
                    return NULL;
                }
            }
            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown preconditioner type!");
            return NULL;
        }
    }
    
    return PyArray_Return(y_vec);
}

/*------------------------------------ GPR Problem Object ------------------------------------*/

static PyObject* GPRProblemObject_loss(GPRProblemObject* self, PyObject* args, PyObject *kwds)
{
    // Parse parameters
    PyArrayObject *pyparams = NULL;
    static char *kwlist[] = {"pyparams", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|", kwlist, &PyArray_Type, &pyparams))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }
    
    // Create a tuple to hold the float values
    PyObject *result_float = NULL;
    switch (self->_dtype)
    {
        case FP64:
        {
            double *pyparams_pt = (double *) PyArray_DATA(pyparams);

            double *loss = (double*) self->_loss;
            double *grad = (double*) self->_grad;
            double *param = (double*) self->_params;
            if (self->_norun)
            {
                self->_norun = 0;
                param[1] = pyparams_pt[0];
                param[2] = pyparams_pt[1];
                param[3] = pyparams_pt[2];
            }
            else
            {
                if (param[1] == pyparams_pt[0] && param[2] == pyparams_pt[1] && param[3] == pyparams_pt[2])
                {
                    result_float = PyFloat_FromDouble(loss[0]);
                    return result_float;
                }
                param[1] = pyparams_pt[0];
                param[2] = pyparams_pt[1];
                param[3] = pyparams_pt[2];
            }

            if (self->_exact_gp)
            {
                exact_gpr_loss_compute( 
                    VAL_TYPE_DOUBLE, self->_transform, self->_krnl, (const void *) param, 
                    self->_n, self->_pt_dim, self->_X_train, self->_n,
                    self->_Y_train, (void *) self->_loss, (void *) self->_grad);
            }
            else
            {
                precond_gpr_loss_compute(
                    (pgp_loss_p) self->_pgp_loss, self->_krnl, (const void *) param, 
                    (void *) self->_loss, (void *) self->_grad, NULL);
            }
            
            result_float = PyFloat_FromDouble(loss[0]);
            break;
        }
        case FP32:
        {
            float *pyparams_pt = (float *) PyArray_DATA(pyparams);

            float *loss = (float*)self->_loss;
            float *grad = (float*)self->_grad;
            float *param = (float*)self->_params;
            if (self->_norun)
            {
                self->_norun = 0;
                param[1] = pyparams_pt[0];
                param[2] = pyparams_pt[1];
                param[3] = pyparams_pt[2];
            }
            else
            {
                if (param[1] == pyparams_pt[0] && param[2] == pyparams_pt[1] && param[3] == pyparams_pt[2])
                {
                    result_float = PyFloat_FromDouble((double)loss[0]);
                    return result_float;
                }
                param[1] = pyparams_pt[0];
                param[2] = pyparams_pt[1];
                param[3] = pyparams_pt[2];
            }

            if (self->_exact_gp)
            {
                exact_gpr_loss_compute( 
                    VAL_TYPE_FLOAT, self->_transform, self->_krnl, (const void *) param, 
                    self->_n, self->_pt_dim, self->_X_train, self->_n,
                    self->_Y_train, (void *) self->_loss, (void *) self->_grad);
            }
            else
            {
                precond_gpr_loss_compute(
                    (pgp_loss_p) self->_pgp_loss, self->_krnl, (const void *) param, 
                    (void *) self->_loss, (void *) self->_grad, NULL);
            }
            
            result_float = PyFloat_FromDouble((double)loss[0]);
            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown data type!");
            return NULL;
        }
    }
    return result_float;
}

static PyObject* GPRProblemObject_grad(GPRProblemObject* self, PyObject* args, PyObject *kwds)
{
    // Parse parameters
    PyArrayObject *pyparams = NULL;
    PyArrayObject *pyparams_out = NULL;
    static char *kwlist[] = {"pyparams", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|", kwlist, &PyArray_Type, &pyparams))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }
    
    // Create a tuple to hold the float values
    PyObject *result_tuple = PyTuple_New(2);

    switch (self->_dtype)
    {
        case FP64:
        {
            npy_intp dim[] = {3};
            pyparams_out = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT64);
            
            double *pyparams_pt = (double *) PyArray_DATA(pyparams);
            double *pyparams_out_pt = (double *) PyArray_DATA(pyparams_out);

            double *loss = (double*) self->_loss;
            double *grad = (double*) self->_grad;
            double *param = (double*) self->_params;
            if (self->_norun)
            {
                self->_norun = 0;
                param[1] = pyparams_pt[0];
                param[2] = pyparams_pt[1];
                param[3] = pyparams_pt[2];
            }
            else
            {
                if (param[1] == pyparams_pt[0] && param[2] == pyparams_pt[1] && param[3] == pyparams_pt[2])
                {
                    PyTuple_SetItem(result_tuple, 0, PyFloat_FromDouble(loss[0]));
                    pyparams_out_pt[0] = grad[0];
                    pyparams_out_pt[1] = grad[1];
                    pyparams_out_pt[2] = grad[2];
                    PyTuple_SetItem(result_tuple, 1, (PyObject*)pyparams_out);
                    return result_tuple;
                }
                param[1] = pyparams_pt[0];
                param[2] = pyparams_pt[1];
                param[3] = pyparams_pt[2];
            }

            if (self->_exact_gp)
            {
                exact_gpr_loss_compute( 
                    VAL_TYPE_DOUBLE, self->_transform, self->_krnl, (const void *) param, 
                    self->_n, self->_pt_dim, self->_X_train, self->_n,
                    self->_Y_train, (void *) self->_loss, (void *) self->_grad);
            }
            else
            {
                precond_gpr_loss_compute(
                    (pgp_loss_p) self->_pgp_loss, self->_krnl, (const void *) param, 
                    (void *) self->_loss, (void *) self->_grad, NULL);
            }
            
            PyTuple_SetItem(result_tuple, 0, PyFloat_FromDouble(loss[0]));
            pyparams_out_pt[0] = grad[0];
            pyparams_out_pt[1] = grad[1];
            pyparams_out_pt[2] = grad[2];
            PyTuple_SetItem(result_tuple, 1, (PyObject*)pyparams_out);
            break;
        }
        case FP32:
        {
            npy_intp dim[] = {3};
            pyparams_out = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT32);
            
            float *pyparams_pt = (float *) PyArray_DATA(pyparams);
            float *pyparams_out_pt = (float *) PyArray_DATA(pyparams_out);

            float *loss = (float*)self->_loss;
            float *grad = (float*)self->_grad;
            float *param = (float*)self->_params;
            if (self->_norun)
            {
                self->_norun = 0;
                param[1] = pyparams_pt[0];
                param[2] = pyparams_pt[1];
                param[3] = pyparams_pt[2];
            }
            else
            {
                if (param[1] == pyparams_pt[0] && param[2] == pyparams_pt[1] && param[3] == pyparams_pt[2])
                {
                    PyTuple_SetItem(result_tuple, 0, PyFloat_FromDouble((double)loss[0]));
                    pyparams_out_pt[0] = grad[0];
                    pyparams_out_pt[1] = grad[1];
                    pyparams_out_pt[2] = grad[2];
                    PyTuple_SetItem(result_tuple, 1, (PyObject*)pyparams_out);
                    return result_tuple;
                }
                param[1] = pyparams_pt[0];
                param[2] = pyparams_pt[1];
                param[3] = pyparams_pt[2];
            }
            if (self->_exact_gp)
            {
                exact_gpr_loss_compute( 
                    VAL_TYPE_FLOAT, self->_transform, self->_krnl, (const void *) param, 
                    self->_n, self->_pt_dim, self->_X_train, self->_n,
                    self->_Y_train, (void *) self->_loss, (void *) self->_grad);
            }
            else
            {
                precond_gpr_loss_compute(
                    (pgp_loss_p) self->_pgp_loss, self->_krnl, (const void *) param,
                    (void *) self->_loss, (void *) self->_grad, NULL);
            }
            
            PyTuple_SetItem(result_tuple, 0, PyFloat_FromDouble((double)loss[0]));
            pyparams_out_pt[0] = grad[0];
            pyparams_out_pt[1] = grad[1];
            pyparams_out_pt[2] = grad[2];
            PyTuple_SetItem(result_tuple, 1, (PyObject*)pyparams_out);
            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown data type!");
            return NULL;
        }
    }
    return result_tuple;
}

static PyObject* GPRProblemObject_is_double(GPRProblemObject* self)
{
    if (self->_dtype == FP64) Py_RETURN_TRUE; 
    else Py_RETURN_FALSE;
}

static PyObject* GPRProblemObject_get_n(GPRProblemObject* self)
{
    return PyLong_FromLong(self->_n);
}

/*------------------------------------ GPC Problem Object ------------------------------------*/

static PyObject* GPCProblemObject_loss(GPCProblemObject* self, PyObject* args, PyObject *kwds)
{
    // Parse parameters
    PyArrayObject *pyparams = NULL;
    static char *kwlist[] = {"pyparams", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|", kwlist, &PyArray_Type, &pyparams))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }
    
    // Create a tuple to hold the float values
    PyObject *result_float = NULL;
    switch (self->_dtype)
    {
        case FP64:
        {
            double *pyparams_pt = (double *) PyArray_DATA(pyparams);

            double *loss = (double*) self->_loss;
            double *grad = (double*) self->_grad;
            double *param = (double*) self->_params;
            if (self->_norun)
            {
                self->_norun = 0;
                for (int i = 0; i < self->_num_classes * 3; i++) 
                    param[i] = pyparams_pt[i];
            }
            else
            {
                int equal = 1;
                for (int i = 0; i < self->_num_classes * 3; i++)
                {
                    if (param[i] != pyparams_pt[i])
                    {
                        equal = 0;
                        break;
                    }
                }

                if (equal)
                {
                    result_float = PyFloat_FromDouble(loss[0]);
                    return result_float;
                }
                else
                {
                    for (int i = 0; i < self->_num_classes * 3; i++) 
                        param[i] = pyparams_pt[i];
                }
            }

            if (self->_exact_gp)
            {
                int val_type = VAL_TYPE_DOUBLE;
                exact_gpc_loss_compute( 
                    val_type, self->_transform, self->_krnl, (const void *) param, 
                    self->_n, self->_pt_dim, self->_X_train, self->_n,
                    self->_Y_train, self->_num_classes, (void *) self->_loss, (void *) self->_grad
                );
            }
            else
            {
                precond_gpc_loss_compute(
                    (pgp_loss_p) self->_pgp_loss, self->_krnl, self->_num_classes, (const void *) param, 
                    (void *) self->_loss, (void *) self->_grad
                );
            }

            result_float = PyFloat_FromDouble(loss[0]);
            break;
        }
        case FP32:
        {
            float *pyparams_pt = (float *) PyArray_DATA(pyparams);

            float *loss = (float*)self->_loss;
            float *grad = (float*)self->_grad;
            float *param = (float*)self->_params;

            if (self->_norun)
            {
                self->_norun = 0;
                for (int i = 0; i < self->_num_classes * 3; i++)
                    param[i] = pyparams_pt[i];
            }
            else
            {
                int equal = 1;
                for (int i = 0; i < self->_num_classes * 3; i++)
                {
                    if (param[i] != pyparams_pt[i])
                    {
                        equal = 0;
                        break;
                    }
                }

                if (equal)
                {
                    result_float = PyFloat_FromDouble((double)loss[0]);
                    return result_float;
                }
                else
                {
                    for (int i = 0; i < self->_num_classes * 3; i++)
                        param[i] = pyparams_pt[i];
                }
            }

            if (self->_exact_gp)
            {
                int val_type = VAL_TYPE_FLOAT;
                exact_gpc_loss_compute( 
                    val_type, self->_transform, self->_krnl, (const void *) param, 
                    self->_n, self->_pt_dim, self->_X_train, self->_n,
                    self->_Y_train, self->_num_classes, (void *) self->_loss, (void *) self->_grad
                );
            }
            else
            {
                precond_gpc_loss_compute(
                    (pgp_loss_p) self->_pgp_loss, self->_krnl, self->_num_classes, (const void *) param, 
                    (void *) self->_loss, (void *) self->_grad
                );
            }

            result_float = PyFloat_FromDouble((double)loss[0]);
            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown data type!");
            return NULL;
        }
    }

    return result_float;
}

static PyObject* GPCProblemObject_grad(GPCProblemObject* self, PyObject* args, PyObject *kwds)
{
    // Parse parameters
    PyArrayObject *pyparams = NULL;
    PyArrayObject *pyparams_out = NULL;
    static char *kwlist[] = {"pyparams", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|", kwlist, &PyArray_Type, &pyparams))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }
    
    // Create a tuple to hold the float values
    PyObject *result_tuple = PyTuple_New(2);
    switch (self->_dtype)
    {
        case FP64:
        {
            npy_intp dim[] = {self->_num_classes * 3};
            pyparams_out = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT64);

            double *pyparams_pt = (double *) PyArray_DATA(pyparams);
            double *pyparams_out_pt = (double *) PyArray_DATA(pyparams_out);

            double *loss = (double*) self->_loss;
            double *grad = (double*) self->_grad;
            double *param = (double*) self->_params;
            if (self->_norun)
            {
                self->_norun = 0;
                for (int i = 0; i < self->_num_classes * 3; i++)
                {
                    param[i] = pyparams_pt[i];
                }
            }
            else
            {
                int equal = 1;
                for (int i = 0; i < self->_num_classes * 3; i++)
                {
                    if (param[i] != pyparams_pt[i])
                    {
                        equal = 0;
                        break;
                    }
                }

                if (equal)
                {
                    PyTuple_SetItem(result_tuple, 0, PyFloat_FromDouble(loss[0]));
                    for (int i = 0; i < self->_num_classes * 3; i++)
                    {
                        pyparams_out_pt[i] = grad[i];
                    }
                    PyTuple_SetItem(result_tuple, 1, (PyObject*)pyparams_out);
                    return result_tuple;
                }
                else
                {
                    for (int i = 0; i < self->_num_classes * 3; i++)
                    {
                        param[i] = pyparams_pt[i];
                    }
                }
            }

            if (self->_exact_gp)
            {
                int val_type = VAL_TYPE_DOUBLE;
                exact_gpc_loss_compute( 
                    val_type, self->_transform, self->_krnl, (const void *) param, 
                    self->_n, self->_pt_dim, self->_X_train, self->_n,
                    self->_Y_train, self->_num_classes, (void *) self->_loss, (void *) self->_grad);
            }
            else
            {
                precond_gpc_loss_compute(
                    (pgp_loss_p) self->_pgp_loss, self->_krnl, self->_num_classes, (const void *) param, 
                    (void *) self->_loss, (void *) self->_grad);
            }
            
            PyTuple_SetItem(result_tuple, 0, PyFloat_FromDouble(loss[0]));
            for (int i = 0; i < self->_num_classes * 3; i++)
            {
                pyparams_out_pt[i] = grad[i];
            }
            PyTuple_SetItem(result_tuple, 1, (PyObject*)pyparams_out);

            break;
        }
        case FP32:
        {
            npy_intp dim[] = {self->_num_classes * 3};
            pyparams_out = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT32);

            float *pyparams_pt = (float *) PyArray_DATA(pyparams);
            float *pyparams_out_pt = (float *) PyArray_DATA(pyparams_out);

            float *loss = (float*)self->_loss;
            float *grad = (float*)self->_grad;
            float *param = (float*)self->_params;

            if (self->_norun)
            {
                self->_norun = 0;
                for (int i = 0; i < self->_num_classes * 3; i++)
                {
                    param[i] = pyparams_pt[i];
                }
            }
            else
            {
                int equal = 1;
                for (int i = 0; i < self->_num_classes * 3; i++)
                {
                    if (param[i] != pyparams_pt[i])
                    {
                        equal = 0;
                        break;
                    }
                }

                if (equal)
                {
                    PyTuple_SetItem(result_tuple, 0, PyFloat_FromDouble((double)loss[0]));
                    for (int i = 0; i < self->_num_classes * 3; i++)
                    {
                        pyparams_out_pt[i] = grad[i];
                    }
                    PyTuple_SetItem(result_tuple, 1, (PyObject*)pyparams_out);
                    return result_tuple;
                }
                else
                {
                    for (int i = 0; i < self->_num_classes * 3; i++)
                    {
                        param[i] = pyparams_pt[i];
                    }
                }
            }

            if (self->_exact_gp)
            {
                int val_type = VAL_TYPE_FLOAT;
                exact_gpc_loss_compute( 
                    val_type, self->_transform, self->_krnl, (const void *) param, 
                    self->_n, self->_pt_dim, self->_X_train, self->_n,
                    self->_Y_train, self->_num_classes, (void *) self->_loss, (void *) self->_grad);
            }
            else
            {
                precond_gpc_loss_compute(
                    (pgp_loss_p) self->_pgp_loss, self->_krnl, self->_num_classes, (const void *) param, 
                    (void *) self->_loss, (void *) self->_grad);
            }

            PyTuple_SetItem(result_tuple, 0, PyFloat_FromDouble((double)loss[0]));
            for (int i = 0; i < self->_num_classes * 3; i++)
            {
                pyparams_out_pt[i] = grad[i];
            }
            PyTuple_SetItem(result_tuple, 1, (PyObject*)pyparams_out);

            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown data type!");
            return NULL;
        }
    }
    return result_tuple;
}

static PyObject* GPCProblemObject_is_double(GPCProblemObject* self)
{
    if (self->_dtype == FP64) Py_RETURN_TRUE;
    else Py_RETURN_FALSE;
}

static PyObject* GPCProblemObject_get_n(GPCProblemObject* self)
{
    return PyLong_FromLong(self->_n);
}

/*===================== Module methods =====================*/

static int krnlmat_init(KrnlMatObject* self, PyObject* args, PyObject* kwds) 
{
    self->_h2 = 1;
    self->_nrow = 0;
    self->_ncol = 0;
    self->_mat = NULL;
    self->_octree = NULL;
    self->_dtype = FP32;
    printf("Init KrnlMatObject done\n");
    return 0;
}

static PyObject* krnlmat_setup(PyObject* self, PyObject* args, PyObject *kwds) 
{
    PyArrayObject *data = NULL;
    PyArrayObject *data2 = NULL;

    // Default parameters
    int    kernel_type  = 1;        // See kernels/kernels.h
    int    use_h2       = 1;        // If we should use H2 matrix
    int    h2_leaf_nmax = 400;      // Maximum number of points in a leaf node
    int    seed         = -1;       // Random seed
    int    nthreads     = -1;       // Max number of OpenMP threads, if -1 we do not apply any change
    double l            = 1.0;      // Lengthscale
    double f            = 1.0;      // Scaling factor
    double s            = 1.0;      // Diagonal shift
    double h2_tol       = 1e-8;     // Accuracy of the H2 matrix
    double h2_leaf_emax = 0;        // Maximum enclosing box size of a leaf node
    
    // Parse parameters
    static char *kwlist[] = {
        "data", "kernel_type", "l", "f", "s", "data2", "use_h2", "nthreads",
        "h2_tol", "h2_leaf_nmax", "h2_leaf_emax", "seed", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!iddd|O!iididi", kwlist,
        &PyArray_Type, &data, &kernel_type, &l, &f, &s, &PyArray_Type, &data2, &use_h2, &nthreads,
        &h2_tol, &h2_leaf_nmax, &h2_leaf_emax, &seed))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }
    
    if (seed > 0)
    {
        srand(seed);
        printf("Change random seed to %d\n",seed);
    }

    if (nthreads > 0)
    {
        omp_set_num_threads(nthreads);
        printf("Change OpenMP threads to %d\n",nthreads);
    }

    // If we have two sets of data, don't use H2 matrix
    if (data2) use_h2 = 0;

    // Each row is a feature, each column is a sample, we use the Fortran column-major style
    int ndim = PyArray_NDIM(data);
    int dim = 1, n = 0;
    if (ndim == 1)
    {
        n = PyArray_DIM(data, 0);
    } else {
        dim = PyArray_DIM(data, 0);
        n = PyArray_DIM(data, 1);
    }

    int dim2 = 1;
    int n2 = 0;
    if (data2)
    {
        int ndim2 = PyArray_NDIM(data2);
        if (ndim2 == 1)
        {
            n2 = PyArray_DIM(data2, 0);
        } else {
            dim2 = PyArray_DIM(data2, 0);
            n2 = PyArray_DIM(data2, 1);
        }
        if (dim != dim2)
        {
            PyErr_SetString(PyExc_ValueError, "Data dimension does not match!");
            return NULL;
        }
        printf("Data2 size %d %d\n", dim2, n2);
    }

    KrnlMatObject* mat_obj = (KrnlMatObject*) KrnlMatObjectType.tp_new(&KrnlMatObjectType, NULL, NULL);
    KrnlMatObjectType.tp_init((PyObject*) mat_obj, NULL, NULL);
    
    mat_obj->_h2    = use_h2;
    mat_obj->_nrow  = n;
    mat_obj->_ncol  = data2 ? n2 : n;
    mat_obj->_dtype = (PyArray_TYPE(data) == NPY_FLOAT32) ? FP32 : FP64;

    // double param[4] = {(double) dim, l, f, s};
    if (use_h2)
    {
        printf("Build H2 Mat dtype %d\n",mat_obj->_dtype);
        printf("Mat size %d %d\n",mat_obj->_nrow,mat_obj->_ncol);
        
        switch (mat_obj->_dtype)
        {
            case FP64:
            {
                int val_type = VAL_TYPE_DOUBLE;
                double *dataset = (double *) PyArray_DATA(data);
                
                octree_p octree = NULL;
                octree_build(n, dim, val_type, dataset, h2_leaf_nmax, (const void *) &h2_leaf_emax, &octree);
                
                ss_h2mat_p ss_h2mat = NULL;
                double param[4] = {(double) dim, l, f, s};
                // Use standard noise level
                ss_h2mat_init(octree, (void *) &param[0], NULL, kernel_type, 1, (void *) &h2_tol, &ss_h2mat);
                
                mat_obj->_mat = (void*)ss_h2mat;
                mat_obj->_octree = (void*)octree;
                break;
            }
            case FP32:
            {
                int val_type = VAL_TYPE_FLOAT;
                float *dataset = (float *) PyArray_DATA(data);
                
                octree_p octree = NULL;
                float h2_leaf_emaxf = (float) h2_leaf_emax;
                octree_build(n, dim, val_type, dataset, h2_leaf_nmax, (const void *) &h2_leaf_emaxf, &octree);
                
                ss_h2mat_p ss_h2mat = NULL;
                float param[4] = {(float) dim, (float) l, (float) f, (float) s};
                float h2_tolf = (float) h2_tol;
                // Use standard noise level
                ss_h2mat_init(octree, (void *) &param[0], NULL, kernel_type, 1, (void *) &h2_tolf, &ss_h2mat);
                
                mat_obj->_mat = (void*)ss_h2mat;
                mat_obj->_octree = (void*)octree;
                break;
            }
            default:
            {
                PyErr_SetString(PyExc_ValueError, "Unknown data type!");
                return NULL;
            }
        }
    }
    else
    {
        printf("Building dense Mat dtype %d\n",mat_obj->_dtype);
        printf("Mat size %d %d\n", mat_obj->_nrow, mat_obj->_ncol);
        
        switch (mat_obj->_dtype)
        {
            case FP64:
            {
                int val_type = VAL_TYPE_DOUBLE;
                double *dataset = (double *) PyArray_DATA(data);
                double *dataset2 = NULL;
                if (data2) dataset2 = (double *) PyArray_DATA(data2);
                
                dense_krnl_mat_p dkmat = NULL;
                double param[4] = {(double) dim, l, f, s};
                // Use standard noise level
                if (data2) dense_krnl_mat_init(n, n, dataset, n2, n2, dataset2, (void *) &param[0], NULL, kernel_type, val_type, &dkmat);
                else       dense_krnl_mat_init(n, n, dataset, n,  n,  dataset,  (void *) &param[0], NULL, kernel_type, val_type, &dkmat);
                
                mat_obj->_mat = (void*) dkmat;
                break;
            }
            case FP32:
            {
                int val_type = VAL_TYPE_FLOAT;
                float *dataset = (float *) PyArray_DATA(data);
                float *dataset2 = NULL;
                if (data2) dataset2 = (float *) PyArray_DATA(data2);
                
                dense_krnl_mat_p dkmat = NULL;
                float param[4] = {(float) dim, (float) l, (float) f, (float) s};
                // use standard noise level
                if (data2) dense_krnl_mat_init(n, n, dataset, n2, n2, dataset2, (void *) &param[0], NULL, kernel_type, val_type, &dkmat);
                else       dense_krnl_mat_init(n, n, dataset, n,  n,  dataset,  (void *) &param[0], NULL, kernel_type, val_type, &dkmat);
                
                mat_obj->_mat = (void*) dkmat;
                break;
            }
            default:
            {
                PyErr_SetString(PyExc_ValueError, "Unknown data type!");
                return NULL;
            }
        }
    }
    
    return (PyObject*) mat_obj;
}

static void krnlmat_dealloc(KrnlMatObject* self) 
{
    if (self->_mat != NULL)
    {
        if (self->_h2)
        {
            ss_h2mat_free((ss_h2mat_p *) &(self->_mat));
            octree_free((octree_p *) &(self->_octree));
            printf("Free H2 matrix done\n");
        }
        else
        {
            dense_krnl_mat_free((dense_krnl_mat_p *) &(self->_mat));
            printf("Free dense kernel matrix done\n");
        }
        self->_mat = NULL;
        self->_octree = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*) self);
    printf("Dealloc KrnlMatObject done\n");
}

static int precond_init(PrecondObject* self, PyObject* args, PyObject* kwds) 
{
    self->_precond_type = 0;
    self->_prec = NULL;
    self->_dtype = FP32;
    printf("Init precond done\n");
    return 0;
}

static PyObject* precond_setup(PyObject* self, PyObject* args, PyObject *kwds)  
{
    PyArrayObject *data = NULL;

    // Default parameters
    int    kernel_type  = 1;        // See kernels/kernels.h
    int    seed         = -1;       // Random seed
    int    nthreads     = -1;       // Max number of OpenMP threads, if -1 we do not apply any change
    int    precond_type = 0;        // Currently only AFN
    int    rank         = 50;       // AFN K11 rank
    int    npt_s        = -rank-1;  // AFN rank estimation sampling size, set to -rank-1 to mute rank estimation
    int    lfil         = 0;        // AFN Schur complement fill level
    double l            = 1.0;      // Lengthscale
    double f            = 1.0;      // Scaling factor
    double s            = 1.0;      // Diagonal shift
    
    // Parse parameters
    static char *kwlist[] = {"data", "kernel_type", "l", "f", "s", "nthreads", "rank", "lfil", "seed", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!iddd|iiii", kwlist,
        &PyArray_Type, &data, &kernel_type, &l, &f, &s, &nthreads, &rank, &lfil, &seed))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }

    if (seed > 0)
    {
        srand(seed);
        printf("Change random seed to %d\n",seed);
    }

    if (nthreads > 0)
    {
        omp_set_num_threads(nthreads);
        printf("Change OpenMP threads to %d\n",nthreads);
    }

    // Each row is a feature, each column is a sample, we use the Fortran column-major style
    int ndim = PyArray_NDIM(data);
    int dim = 1, n = 0;
    if (ndim == 1)
    {
        n = PyArray_DIM(data, 0);
    } else {
        dim = PyArray_DIM(data, 0);
        n = PyArray_DIM(data, 1);
    }
    if (n < npt_s) npt_s = n;

    PrecondObject* precond_obj = (PrecondObject*) PrecondObjectType.tp_new(&PrecondObjectType, NULL, NULL);
    PrecondObjectType.tp_init((PyObject*)precond_obj, NULL, NULL);
    
    precond_obj->_n = n;
    precond_obj->_precond_type = 0; // Currently only AFN
    precond_obj->_dtype = (PyArray_TYPE(data) == NPY_FLOAT32) ? FP32 : FP64;

    krnl_func krnl;
    int krnl_id = 1;
    int need_grad = 1;
    switch (kernel_type)
    {
        case 1:
        {
            krnl_id = KERNEL_ID_GAUSSIAN;
            break;
        }
        case 2: 
        {
            krnl_id = KERNEL_ID_MATERN32;
            break;
        }
        case 3: 
        {
            krnl_id = KERNEL_ID_MATERN52;
            break;
        }
        case 99: 
        {
            krnl_id = KERNEL_ID_CUSTOM;
            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown kernel type!");
            return NULL;
        }
    }

    switch (precond_obj->_precond_type)
    {
        case 0:
        {
            // AFN preconditioner
            printf("Build AFN preconditioner dtype %d\n",precond_obj->_dtype);
            
            switch (precond_obj->_dtype)
            {
                case FP64:
                {
                    int val_type = VAL_TYPE_DOUBLE;
                    double *dataset = (double *) PyArray_DATA(data);
                    
                    // TODO: if use H2 matrix, can pass the octree struct to afn_precond_build
                    afn_precond_p ap = NULL;
                    double param[4] = {(double) dim, l, f, s};
                    // The stand-alone AFN does not support dnoise
                    afn_precond_build(
                        val_type, krnl_id, (void *) &param[0], NULL,
                        n, dim, (const void *) dataset, n, 
                        npt_s, rank, lfil, 
                        NULL, need_grad, &ap
                    );

                    precond_obj->_prec = (void*) ap;
                    break;
                }
                case FP32:
                {
                    int val_type = VAL_TYPE_FLOAT;
                    float *dataset = (float *) PyArray_DATA(data);
                    
                    afn_precond_p ap;
                    float param[4] = {(float) dim, (float) l, (float) f, (float) s};
                    // The stand-alone AFN does not support dnoise
                    afn_precond_build(
                        val_type, krnl_id, (void *) &param[0], NULL,
                        n, dim, (const void *) dataset, n, 
                        npt_s, rank, lfil, 
                        NULL, need_grad, &ap
                    );
                    
                    precond_obj->_prec = (void*) ap;
                    break;
                }
                default:
                {
                    PyErr_SetString(PyExc_ValueError, "Unknown data type!");
                    return NULL;
                }
            }
            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown preconditioner type!");
            return NULL;
        }
    }
    
    return (PyObject*) precond_obj;
}

static void precond_dealloc(PrecondObject* self) 
{
    if (self->_prec != NULL)
    {
        switch (self->_precond_type)
        {
            case 0:
            {
                afn_precond_free((afn_precond_p *) &(self->_prec));
                self->_prec = NULL;
                break;
            }
            default:
            {
                PyErr_SetString(PyExc_ValueError, "Unknown preconditioner type!");
            }
        }
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
    printf("Dealloc precond done\n");
}

static int gpr_problem_init(GPRProblemObject* self, PyObject* args, PyObject* kwds)
{
    self->_n         = 0;
    self->_krnl      = 1;
    self->_transform = 0;
    self->_dtype     = FP32;

    self->_exact_gp  = 0;
    self->_pt_dim    = 0;
    self->_X_train   = NULL;
    self->_Y_train   = NULL;

    self->_params    = NULL;
    self->_loss      = NULL;
    self->_grad      = NULL;
    self->_norun     = 1;

    self->_pgp_loss  = NULL;

    printf("Init gpr problem done\n");
    return 0;
}

int parse_gp_params(
    const int kernel_type, int *_krnl, const int transform, int *nnt_id,
    const int npt, const int dim, const int mvtype, int *kmat_alg, 
    const int print_info, const int dtype, const int exact_gp, const int rank, const int lfil
)
{
    int ret = 1;

    switch (kernel_type)
    {
        case 1:  *_krnl = KERNEL_ID_GAUSSIAN; break;
        case 2:  *_krnl = KERNEL_ID_MATERN32; break;
        case 3:  *_krnl = KERNEL_ID_MATERN52; break;
        case 99: *_krnl = KERNEL_ID_CUSTOM;   break;
        default: 
        {
            PyErr_SetString(PyExc_ValueError, "Unknown kernel type!");
            ret = 0;
        }
    }

    switch (transform)
    {
        case 0: *nnt_id = NNT_SOFTPLUS; break;
        case 1: *nnt_id = NNT_EXP;      break;
        case 2: *nnt_id = NNT_SIGMOID;  break;
        default: 
        {
            PyErr_SetString(PyExc_ValueError, "Unknown transform type!");
            ret = 0;
        }
    }

    switch (mvtype)
    {
        case 1: *kmat_alg = SYMM_KMAT_ALG_DENSE_FORM; break;
        case 2: *kmat_alg = SYMM_KMAT_ALG_DENSE;      break;
        case 0: default:
        {
            if ((npt < _h2_min_size) || (dim > 3)) *kmat_alg = SYMM_KMAT_ALG_DENSE_FORM;
            else *kmat_alg = SYMM_KMAT_ALG_H2_PROXY;
        }
    }

    // Below are print statements
    if (!print_info) return ret;

    switch (dtype)
    {
        case FP64: printf("Data type: double\n"); break;
        case FP32: printf("Data type: float\n");  break;
        default: 
        {
            PyErr_SetString(PyExc_ValueError, "Unknown data type!");
            ret = 0;
        }
    }

    switch (kernel_type)
    {
        case 1:  printf("Kernel type: Gaussian\n");   break;
        case 2:  printf("Kernel type: Matern 3/2\n"); break;
        case 3:  printf("Kernel type: Matern 5/2\n"); break;
        case 99: printf("Kernel type: custom\n");     break;
        default: printf("Unknown kernel type!\n");
    }

    switch (transform)
    {
        case 0: printf("Transform type: softplus\n"); break;
        case 1: printf("Transform type: exp\n");      break;
        case 2: printf("Transform type: sigmoid\n");  break;
        default: printf("Unknown transform type!\n");
    }

    if (exact_gp)
    {
        printf("Using exact GP\n");
    } else {
        printf("Using preconditioned GP, kernel matrix form: ");
        switch (*kmat_alg)
        {
            case SYMM_KMAT_ALG_H2_PROXY: printf("H2 matrix\n"); break;
            case SYMM_KMAT_ALG_DENSE_FORM: printf("dense / fall back to on-the-fly\n"); break;
            case SYMM_KMAT_ALG_DENSE: printf("on-the-fly\n"); break;
        }
        printf("AFN preconditioner parameters: rank %d, lfil %d\n", rank, lfil);
    }

    return ret;
}

static PyObject* gpr_problem_setup(PyObject* self, PyObject* args, PyObject *kwds)
{
    PyArrayObject *data = NULL;
    PyArrayObject *label = NULL;

    // Default parameters
    int kernel_type = 1;        // See kernels/kernels.h
    int transform   = 0;        // 0: softplus; 1: exp; 2: sigmoid
    int exact_gp    = 0;
    int mvtype      = 0;        // 0: default. H2 when possible, fall back to dense or on-the-fly; 1: dense or on-the-fly; 2: on-the-fly
    int rank        = 50;       // AFN K11 rank
    int npt_s       = -rank-1;  // AFN rank estimation sampling size, set to -rank-1 to mute rank estimation
    int lfil        = 0; 
    int niter       = 10;
    int nvec        = 10;
    int nthreads    = -1;       // Max number of OpenMP threads, if -1 we do not apply any change
    int seed        = -1;       // Random seed
    
    // Parse parameters
    static char *kwlist[] = {
        "data", "label", "kernel_type", "nthreads", "exact_gp",
        "mvtype", "rank", "lfil", "niter", "nvec", "seed", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!i|iiiiiiii", kwlist,
        &PyArray_Type, &data, &PyArray_Type, &label, &kernel_type, &nthreads, &exact_gp, 
        &mvtype, &rank, &lfil, &niter, &nvec, &seed))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }
    
    if (nthreads > 0)
    {
        omp_set_num_threads(nthreads);
        printf("Change OpenMP threads to %d\n",nthreads);
    }

    if (seed > 0)
    {
        srand(seed);
        printf("Change random seed to %d\n",seed);
    }

    // Each row is a feature, each column is a sample, we use the Fortran column-major style
    int ndim = PyArray_NDIM(data);
    int dim = 1, n = 0;
    if (ndim == 1)
    {
        n = PyArray_DIM(data, 0);
    } else {
        dim = PyArray_DIM(data, 0);
        n = PyArray_DIM(data, 1);
    }
    if (n < npt_s) npt_s = n;

    printf("Setting up GPR problem with %d samples and %d features\n",n,dim);

    GPRProblemObject* gprproblem_obj = (GPRProblemObject*) GPRProblemObjectType.tp_new(&GPRProblemObjectType, NULL, NULL);

    GPRProblemObjectType.tp_init((PyObject*) gprproblem_obj, NULL, NULL);

    gprproblem_obj->_n = n;
    gprproblem_obj->_pt_dim = dim;
    gprproblem_obj->_dtype = (PyArray_TYPE(data) == NPY_FLOAT32) ? FP32 : FP64;
    gprproblem_obj->_exact_gp = exact_gp? 1 : 0;

    int nnt_id = 0, kmat_alg = 0;
    printf("====================================\n");
    printf("Creating GPR problem\n");
    int ret = parse_gp_params(
        kernel_type, &gprproblem_obj->_krnl, transform, &nnt_id,
        n, dim, mvtype, &kmat_alg, 
        1, gprproblem_obj->_dtype, exact_gp, rank, lfil
    );
    printf("LanQ parameters: niter %d, nvec %d\n", niter, nvec);
    printf("====================================\n");
    gprproblem_obj->_transform = nnt_id;
    if (lfil > 0) npt_s = -(rank+1);  // Mute rank estimation and use AFN
    else npt_s = -rank;               // Mute rank estimation and use Nystrom
    if (ret == 0) return NULL;

    if (exact_gp)
    {
        // Need to store X and Y
        if (gprproblem_obj->_dtype == FP64)
        {
            gprproblem_obj->_X_train = (double *) malloc(n * dim * sizeof(double));
            gprproblem_obj->_Y_train = (double *) malloc(n * sizeof(double));
            double *x_data = (double *) PyArray_DATA(data);
            double *y_label = (double *) PyArray_DATA(label);
            memcpy(gprproblem_obj->_X_train, x_data, n * dim * sizeof(double));
            memcpy(gprproblem_obj->_Y_train, y_label, n * sizeof(double));
        }
        else
        {
            gprproblem_obj->_X_train = (float *) malloc(n * dim * sizeof(float));
            gprproblem_obj->_Y_train = (float *) malloc(n * sizeof(float));
            float *x_data = (float *) PyArray_DATA(data);
            float *y_label = (float *) PyArray_DATA(label);
            memcpy(gprproblem_obj->_X_train, x_data, n * dim * sizeof(float));
            memcpy(gprproblem_obj->_Y_train, y_label, n * sizeof(float));
        }
    }

    switch (gprproblem_obj->_dtype)
    {
        case FP64:
        {
            int val_type = VAL_TYPE_DOUBLE;
            double *x_data = (double *) PyArray_DATA(data);
            double *y_label = (double *) PyArray_DATA(label);

            if (!gprproblem_obj->_exact_gp)
            {
                // Task only decides data type
                pgp_loss_init(
                    val_type, nnt_id, n, dim, 
                    x_data, n, y_label, val_type,
                    npt_s, rank, lfil, niter, 
                    nvec, kmat_alg, (pgp_loss_p *) &(gprproblem_obj->_pgp_loss)
                );
            }

            gprproblem_obj->_loss = malloc(sizeof(double));
            gprproblem_obj->_grad = malloc(sizeof(double) * 3);
            gprproblem_obj->_params = malloc(sizeof(double) * 4);
            double *param = (double *) gprproblem_obj->_params;
            param[0] = (double) dim;

            break;
        }
        case FP32:
        {
            int val_type = VAL_TYPE_FLOAT;
            float *x_data = (float *) PyArray_DATA(data);
            float *y_label = (float *) PyArray_DATA(label);

            if (!gprproblem_obj->_exact_gp)
            {
                // task only decides data type
                pgp_loss_init(
                    val_type, nnt_id, n, dim, 
                    x_data, n, y_label, val_type,
                    npt_s, rank, lfil, niter, 
                    nvec, kmat_alg, (pgp_loss_p *) &(gprproblem_obj->_pgp_loss)
                );
            }

            gprproblem_obj->_loss = malloc(sizeof(float));
            gprproblem_obj->_grad = malloc(sizeof(float) * 3);
            gprproblem_obj->_params = malloc(sizeof(float) * 4);
            float *param = (float *) gprproblem_obj->_params;
            param[0] = (float) dim;

            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown data type!");
            return NULL;
        }
    }
    
    return (PyObject*) gprproblem_obj;
}

static void gpr_problem_dealloc(GPRProblemObject* self)
{
    if (self->_pgp_loss != NULL)
    {
        pgp_loss_free((pgp_loss_p *) &(self->_pgp_loss));
        self->_pgp_loss = NULL;
    }
    if (self->_X_train != NULL)
    {
        free(self->_X_train);
        self->_X_train = NULL;
    }
    if (self->_Y_train != NULL)
    {
        free(self->_Y_train);
        self->_Y_train = NULL;
    }
    if (self->_params != NULL)
    {
        free(self->_params);
        self->_params = NULL;
    }
    if (self->_loss != NULL)
    {
        free(self->_loss);
        self->_loss = NULL;
    }
    if (self->_grad != NULL)
    {
        free(self->_grad);
        self->_grad = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*) self);
    printf("Dealloc gpr problem done\n");
}

static int gpc_problem_init(GPCProblemObject* self, PyObject* args, PyObject* kwds)
{
    self->_pgp_loss = NULL;

    self->_n            = 0;
    self->_krnl         = 1;
    self->_transform    = 0;        // 0: softmax
    self->_dtype        = FP32;
    self->_num_classes  = 1;        // GP_REGRESSION: 1, GP_CLASSIFICATION: >= 1

    self->_exact_gp = 0;
    self->_pt_dim   = 0;
    self->_X_train  = NULL;
    self->_Y_train  = NULL;
    
    self->_params   = NULL;
    self->_loss     = NULL;
    self->_grad     = NULL;
    self->_norun    = 1;

    self->_pgp_loss = NULL;

    printf("Init gpc problem done\n");
    return 0;
}

static PyObject* gpc_problem_setup(PyObject* self, PyObject* args, PyObject *kwds)
{
    PyArrayObject *data = NULL;
    PyArrayObject *label = NULL;
    
    int kernel_type = 1;        // See kernels/kernels.h
    int transform   = 0;        // 0: softmax, 1: exp, 2: sigmoid
    int exact_gp    = 0;
    int mvtype      = 0;        // 0: default. H2 when possible, otherwise dense; 1: dense or on-the-fly; 2: on-the-fly
    int rank        = 50;
    int npt_s       = -rank-1;  // AFN rank estimation sampling size, set to -rank-1 to mute rank estimation
    int lfil        = 0;
    int niter       = 10;
    int nvec        = 10;
    int nthreads    = -1;       // Max number of OpenMP threads, if -1 we do not apply any change
    int seed        = -1;       // Random seed
    
    // Parse parameters
    static char *kwlist[] = {
        "data", "label", "kernel_type", "nthreads", "exact_gp",
        "mvtype", "rank", "lfil", "niter", "nvec", "seed", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!i|iiiiiiii", kwlist,
        &PyArray_Type, &data, &PyArray_Type, &label, &kernel_type, &nthreads, &exact_gp, 
        &mvtype, &rank, &lfil, &niter, &nvec, &seed))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }
    
    if (nthreads > 0)
    {
        omp_set_num_threads(nthreads);
        printf("Change OpenMP threads to %d\n",nthreads);
    }

    if (seed > 0)
    {
        srand(seed);
        printf("Change random seed to %d\n",seed);
    }

    // Each row is a feature, each column is a sample, we use the Fortran column-major style
    int ndim = PyArray_NDIM(data);
    int dim = 1, n = 0;
    if (ndim == 1)
    {
        n = PyArray_DIM(data, 0);
    } else {
        dim = PyArray_DIM(data, 0);
        n = PyArray_DIM(data, 1);
    }
    if (n < npt_s) npt_s = n;

    printf("Setting up GPC problem with %d samples and %d features\n",n,dim);

    GPCProblemObject* gpcproblem_obj = (GPCProblemObject*) GPCProblemObjectType.tp_new(&GPCProblemObjectType, NULL, NULL);

    GPCProblemObjectType.tp_init((PyObject*)gpcproblem_obj, NULL, NULL);

    gpcproblem_obj->_n = n;
    gpcproblem_obj->_pt_dim = dim;
    gpcproblem_obj->_exact_gp = exact_gp ? 1 : 0;
    gpcproblem_obj->_dtype = (PyArray_TYPE(data) == NPY_FLOAT32) ? FP32 : FP64;

    // Search for the number of classes,
    // we assume that the label is a between 0 and num_classes - 1
    int num_classes = 0;
    for (int i = 0; i < n; i++)
    {
        PyObject *item = PyArray_GETITEM(label, PyArray_GETPTR1(label, i));
        long result = PyLong_AsLong(item);
        Py_DECREF(item);
        if (result > num_classes) num_classes = (int) result;
    }
    num_classes++;
    gpcproblem_obj->_num_classes = num_classes;

    if (num_classes < 2)
    {
        PyErr_SetString(PyExc_ValueError, "Number of classes should be at least 2!");
        return NULL;
    }
    
    if (exact_gp)
    {
        // Need to store X and Y
        if (gpcproblem_obj->_dtype == FP64)
        {
            gpcproblem_obj->_X_train = (double *) malloc(n * dim * sizeof(double));
            double *x_data = (double *) PyArray_DATA(data);
            memcpy(gpcproblem_obj->_X_train, x_data, n * dim * sizeof(double));
        }
        else
        {
            gpcproblem_obj->_X_train = (float *) malloc(n * dim * sizeof(float));
            float *x_data = (float *) PyArray_DATA(data);
            memcpy(gpcproblem_obj->_X_train, x_data, n * dim * sizeof(float));
        }
        // Store Y as integer type
        int *Y_train = (int *) malloc(n * sizeof(int));
        for (int i = 0; i < n; i++)
        {
            PyObject *item = PyArray_GETITEM(label, PyArray_GETPTR1(label, i));
            long result = PyLong_AsLong(item);
            Py_DECREF(item);
            Y_train[i] = (int)result;
        }
        gpcproblem_obj->_Y_train = (void*)Y_train;
    }
    else
    {
        // Need to store Y temporarily
        int *Y_train = (int *) malloc(n * sizeof(int));
        for (int i = 0; i < n; i++)
        {
            PyObject *item = PyArray_GETITEM(label, PyArray_GETPTR1(label, i));
            long result = PyLong_AsLong(item);
            Py_DECREF(item);
            Y_train[i] = (int)result;
        }
        gpcproblem_obj->_Y_train = (void*)Y_train;
    }

    int nnt_id = 0, kmat_alg = 0;
    printf("====================================\n");
    printf("Creating GPC problem\n");
    printf("Number of classes: %d\n",gpcproblem_obj->_num_classes);
    int ret = parse_gp_params(
        kernel_type, &gpcproblem_obj->_krnl, transform, &nnt_id,
        n, dim, mvtype, &kmat_alg, 
        1, gpcproblem_obj->_dtype, exact_gp, rank, lfil
    );
    printf("LanQ parameters: niter %d, nvec %d\n",niter,nvec);
    printf("====================================\n");
    gpcproblem_obj->_transform = nnt_id;
    if (lfil > 0) npt_s = -(rank+1);  // Mute rank estimation and use AFN
    else npt_s = -rank;               // Mute rank estimation and use Nystrom
    if (ret == 0) return NULL;

    switch (gpcproblem_obj->_dtype)
    {
        case FP64:
        {
            int val_type = VAL_TYPE_DOUBLE;
            double *x_data = (double *) PyArray_DATA(data);

            if (!gpcproblem_obj->_exact_gp)
            {
                // Task only decides data type
                pgp_loss_init(
                    val_type, nnt_id, n, dim, 
                    x_data, n, gpcproblem_obj->_Y_train, VAL_TYPE_INT,
                    npt_s, rank, lfil, niter,
                    nvec, kmat_alg, (pgp_loss_p *) &(gpcproblem_obj->_pgp_loss)
                );
            }

            gpcproblem_obj->_loss = malloc(sizeof(double));
            gpcproblem_obj->_grad = malloc(sizeof(double) * 3 * gpcproblem_obj->_num_classes);
            // For regression 4 is needed, for classification num_classes * 3 is needed
            gpcproblem_obj->_params = malloc(sizeof(double) * (3 * gpcproblem_obj->_num_classes));
            double *param = (double *) gpcproblem_obj->_params;

            break;
        }
        case FP32:
        {
            int val_type = VAL_TYPE_FLOAT;
            float *x_data = (float *) PyArray_DATA(data);
            float *y_label = (float *) PyArray_DATA(label);

            if (!gpcproblem_obj->_exact_gp)
            {
                // Task only decides data type
                pgp_loss_init(
                    val_type, nnt_id, n, dim, 
                    x_data, n, gpcproblem_obj->_Y_train, VAL_TYPE_INT,
                    npt_s, rank, lfil, niter, 
                    nvec, kmat_alg, (pgp_loss_p *) &(gpcproblem_obj->_pgp_loss)
                );
            }

            gpcproblem_obj->_loss = malloc(sizeof(float));
            gpcproblem_obj->_grad = malloc(sizeof(float) * 3 * gpcproblem_obj->_num_classes);
            // For regression 4 is needed, for classification num_classes * 3 is needed
            gpcproblem_obj->_params = malloc(sizeof(float) * (3 * gpcproblem_obj->_num_classes));
            float *param = (float *) gpcproblem_obj->_params;

            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown data type!");
            return NULL;
        }
    }
    
    return (PyObject*) gpcproblem_obj;
}

static void gpc_problem_dealloc(GPCProblemObject* self)
{
    if (self->_pgp_loss != NULL)
    {
        pgp_loss_free((pgp_loss_p *) &(self->_pgp_loss));
        self->_pgp_loss = NULL;
    }
    if (self->_X_train != NULL)
    {
        free(self->_X_train);
        self->_X_train = NULL;
    }
    if (self->_Y_train != NULL)
    {
        free(self->_Y_train);
        self->_Y_train = NULL;
    }
    if (self->_params != NULL)
    {
        free(self->_params);
        self->_params = NULL;
    }
    if (self->_loss != NULL)
    {
        free(self->_loss);
        self->_loss = NULL;
    }
    if (self->_grad != NULL)
    {
        free(self->_grad);
        self->_grad = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*) self);
    printf("Dealloc gpc problem done\n");
}

/*------------------------------------HiGP Module------------------------------------*/

PyMODINIT_FUNC PyInit_higp_cext(void) 
{
    import_array();

    if (PyType_Ready(&KrnlMatObjectType) < 0 || 
        PyType_Ready(&PrecondObjectType) < 0 || 
        PyType_Ready(&GPRProblemObjectType) < 0 ||
        PyType_Ready(&GPCProblemObjectType) < 0
        )
        return NULL;

    PyObject *higp_cext_ = PyModule_Create(&higp_cext_module);
    if (!higp_cext_) return NULL;

    PyObject *krnlmat = PyModule_Create(&krnlmatmodule);
    if (!krnlmat) return NULL;

    PyObject *precond = PyModule_Create(&precondmodule);
    if (!precond) return NULL;

    PyObject *gprproblem = PyModule_Create(&gprproblemmodule);
    if (!gprproblem) return NULL;

    PyObject *gpcproblem = PyModule_Create(&gpcproblemmodule);
    if (!gpcproblem) return NULL;

    Py_INCREF(&KrnlMatObjectType);
    PyModule_AddObject(krnlmat, "KrnlMatObject", (PyObject *)&KrnlMatObjectType);

    Py_INCREF(&PrecondObjectType);
    PyModule_AddObject(precond, "PrecondObject", (PyObject *)&PrecondObjectType);

    Py_INCREF(&GPRProblemObjectType);
    PyModule_AddObject(gprproblem, "GPRProblemObject", (PyObject *)&GPRProblemObjectType);

    Py_INCREF(&GPCProblemObjectType);
    PyModule_AddObject(gpcproblem, "GPCProblemObject", (PyObject *)&GPCProblemObjectType);

    Py_INCREF(krnlmat);
    PyModule_AddObject(higp_cext_, "krnlmat", krnlmat);

    Py_INCREF(precond);
    PyModule_AddObject(higp_cext_, "precond", precond);

    Py_INCREF(gprproblem);
    PyModule_AddObject(higp_cext_, "gprproblem", gprproblem);

    Py_INCREF(gpcproblem);
    PyModule_AddObject(higp_cext_, "gpcproblem", gpcproblem);

    return higp_cext_;
}

static PyObject* HiGP_Cext_gpr_prediction(PyObject* self, PyObject *args, PyObject *kwds)
{
    PyArrayObject *pyparams = NULL;
    PyArrayObject *data_train = NULL;
    PyArrayObject *label_train = NULL;
    PyArrayObject *data_prediction = NULL;
    PyArrayObject *label_prediction = NULL;
    PyArrayObject *stddev = NULL;

    int    kernel_type = 1;         // See kernels/kernels.h
    int    transform   = 0;         // 0: softmax; 1: exp; 2: sigmoid
    int    exact_gp    = 0;
    int    mvtype      = 0;         // 0: default. H2 when possible, otherwise dense; 1: dense on-the-fly; 2: dense
    int    rank        = 50;
    int    npt_s       = -rank-1;   // AFN rank estimation sampling size, set to -rank-1 to mute rank estimation
    int    lfil        = 0;
    int    niter       = 50;
    int    nthreads    = -1;        // Max number of OpenMP threads, if -1 we do not apply any change
    double tol         = 1e-6;
    
    // Parse parameters
    static char *kwlist[] = {
        "data_train", "label_train", "data_prediction", "kernel_type", "pyparams",
        "nthreads", "exact_gp", "mvtype", "rank", "lfil", "niter", "tol", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!iO!|iiiiiid", kwlist,
        &PyArray_Type, &data_train, &PyArray_Type, &label_train, &PyArray_Type, &data_prediction, &kernel_type, &PyArray_Type, 
        &pyparams, &nthreads, &exact_gp, &mvtype, &rank, &lfil, &niter, &tol))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }

    if (nthreads > 0)
    {
        omp_set_num_threads(nthreads);
        printf("Change OpenMP threads to %d\n",nthreads);
    }

    // Each row is a feature, each column is a sample, we use the Fortran column-major style
    int ndim_train = PyArray_NDIM(data_train);
    int ndim_pred = PyArray_NDIM(data_prediction);
    int dim = 1, n1 = 0, n2 = 0;
    if (ndim_train == 1)
    {
        n1 = PyArray_DIM(data_train, 0);
    } else {
        dim = PyArray_DIM(data_train, 0);
        n1 = PyArray_DIM(data_train, 1);
    }
    if (ndim_pred == 1)
    {
        n2 = PyArray_DIM(data_prediction, 0);
    } else {
        n2 = PyArray_DIM(data_prediction, 1);
    }
    if (n1 < npt_s) npt_s = n1;

    // Determine the data type
    dtype_enum dtype = (PyArray_TYPE(data_train) == NPY_FLOAT32) ? FP32 : FP64;
    dtype_enum dtype2 = (PyArray_TYPE(data_prediction) == NPY_FLOAT32) ? FP32 : FP64;
    if (dtype != dtype2)
    {
        PyErr_SetString(PyExc_ValueError, "Data type does not match!");
        return NULL;
    }

    int krnl = 1, nnt_id = 0, kmat_alg = 0;
    printf("====================================\n");
    printf("Running GPR prediction\n");
    int ret = parse_gp_params(
        kernel_type, &krnl, transform, &nnt_id,
        n1, dim, mvtype, &kmat_alg, 
        1, dtype, exact_gp, rank, lfil
    );
    printf("PCG parameters: niter %d, tol %g\n", niter, tol);
    printf("====================================\n");
    if (lfil > 0) npt_s = -(rank+1);  // Mute rank estimation and use AFN
    else npt_s = -rank;               // Mute rank estimation and use Nystrom
    if (ret == 0) return NULL;
    
    // Determine the return type
    switch (dtype)
    {
        case FP64:
        {
            npy_intp dim[] = {n2};
            label_prediction = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT64);
            stddev           = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT64);
            break;
        }
        case FP32:
        {
            npy_intp dim[] = {n2};
            label_prediction = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT32);
            stddev           = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT32);
            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown data type!");
            return NULL;
        }
    }

    switch (dtype)
    {
        case FP64:
        {
            int val_type = VAL_TYPE_DOUBLE;
            double *pyparams_pt = (double *) PyArray_DATA(pyparams);
            double *X_train = (double *) PyArray_DATA(data_train);
            double *Y_train = (double *) PyArray_DATA(label_train);
            double *X_pred = (double *) PyArray_DATA(data_prediction);
            double *Y_pred = (double *) PyArray_DATA(label_prediction);
            double *Y_stddev = (double *) PyArray_DATA(stddev);

            double param[4] = {(double) dim, pyparams_pt[0], pyparams_pt[1], pyparams_pt[2]};

            if (exact_gp)
            {
                exact_gpr_predict(
                    val_type, nnt_id, krnl, (const void *) &param[0],
                    n1, dim, X_train, n1,
                    Y_train, n2, X_pred, n2,
                    Y_pred, Y_stddev
                );
            }
            else
            {
                precond_gpr_predict(
                    val_type, nnt_id, krnl, (const void *) &param[0],
                    n1, dim, X_train, n1,
                    Y_train, n2, X_pred, n2,
                    npt_s, rank, lfil, niter, 
                    (const void *) &tol, kmat_alg, Y_pred, Y_stddev
                );
            }

            break;
        }
        case FP32:
        {
            int val_type = VAL_TYPE_FLOAT;
            float *pyparams_pt = (float *) PyArray_DATA(pyparams);
            float *X_train = (float *) PyArray_DATA(data_train);
            float *Y_train = (float *) PyArray_DATA(label_train);
            float *X_pred = (float *) PyArray_DATA(data_prediction);
            float *Y_pred = (float *) PyArray_DATA(label_prediction);
            float *Y_stddev = (float *) PyArray_DATA(stddev);
            float param[4] = {(float) dim, pyparams_pt[0], pyparams_pt[1], pyparams_pt[2]};
            float tol_f = tol > 1e-5 ? tol : 1e-5;

            if (exact_gp)
            {
                exact_gpr_predict(
                    val_type, nnt_id, krnl, &param[0], 
                    n1, dim, X_train, n1, 
                    Y_train, n2, X_pred, n2, 
                    Y_pred, Y_stddev
                );
            }
            else
            {
                precond_gpr_predict(
                    val_type, nnt_id, krnl, &param[0], 
                    n1, dim, X_train, n1, 
                    Y_train, n2, X_pred, n2, 
                    npt_s, rank, lfil, niter, 
                    (const void *) &tol_f, kmat_alg, Y_pred, Y_stddev
                );
            }

            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown data type!");
            return NULL;
        }
    }
    
    PyObject* results = PyTuple_Pack(2, label_prediction, stddev);
    return results;
}

static PyObject* HiGP_Cext_gpc_prediction(PyObject* self, PyObject *args, PyObject *kwds)
{
    PyArrayObject *pyparams = NULL;
    PyArrayObject *data_train = NULL;
    PyArrayObject *label_train = NULL;
    PyArrayObject *data_prediction = NULL;
    PyArrayObject *label_prediction = NULL;
    PyArrayObject *label_value = NULL;
    PyArrayObject *label_prob = NULL;

    int    kernel_type = 1;         // See kernels/kernels.h
    int    transform   = 0;         // 0: softmax; 1: exp; 2: sigmoid
    int    exact_gp    = 0;
    int    mvtype      = 0;         // 0: default. H2 when possible, otherwise dense; 1: dense on-the-fly; 2: dense
    int    rank        = 50;
    int    npt_s       = -rank-1;   // AFN rank estimation sampling size, set to -rank-1 to mute rank estimation
    int    lfil        = 0;
    int    niter       = 50;
    int    nsample     = 256;
    int    nthreads    = -1;        // Max number of OpenMP threads, if -1 we do not apply any change
    double tol         = 1e-6;

    // Parse parameters
    static char *kwlist[] = {
        "data_train", "label_train", "data_prediction", "kernel_type", "pyparams", 
        "nthreads", "exact_gp", "mvtype", "nsample", "rank", "lfil", "niter", "tol", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!iO!|iiiiiiid", kwlist,
        &PyArray_Type, &data_train, &PyArray_Type, &label_train, &PyArray_Type, &data_prediction, &kernel_type, &PyArray_Type, &pyparams,
        &nthreads, &exact_gp, &mvtype, &nsample, &rank, &lfil, &niter, &tol))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }

    if (nthreads > 0)
    {
        omp_set_num_threads(nthreads);
        printf("Change OpenMP threads to %d\n",nthreads);
    }

    // Each row is a feature, each column is a sample, we use the Fortran column-major style
    int ndim_train = PyArray_NDIM(data_train);
    int ndim_pred = PyArray_NDIM(data_prediction);
    int dim = 1, n1 = 0, n2 = 0;
    if (ndim_train == 1)
    {
        n1 = PyArray_DIM(data_train, 0);
    } else {
        dim = PyArray_DIM(data_train, 0);
        n1 = PyArray_DIM(data_train, 1);
    }
    if (ndim_pred == 1)
    {
        n2 = PyArray_DIM(data_prediction, 0);
    } else {
        n2 = PyArray_DIM(data_prediction, 1);
    }
    if (n1 < npt_s) npt_s = n1;

    // Determine the data type
    dtype_enum dtype = (PyArray_TYPE(data_train) == NPY_FLOAT32) ? FP32 : FP64;
    dtype_enum dtype2 = (PyArray_TYPE(data_prediction) == NPY_FLOAT32) ? FP32 : FP64;
    if (dtype != dtype2)
    {
        PyErr_SetString(PyExc_ValueError, "Data type does not match!");
        return NULL;
    }

    int num_classes = 0;
    int *training_label_buffer = (int *) malloc(n1 * sizeof(int));
    for (int i = 0; i < n1; i++)
    {
        PyObject *item = PyArray_GETITEM(label_train, PyArray_GETPTR1(label_train, i));
        long result = PyLong_AsLong(item);
        Py_DECREF(item);
        if (result > num_classes) num_classes = (int)result;
        training_label_buffer[i] = (int)result;
    }
    num_classes++;
    if (num_classes < 2)
    {
        PyErr_SetString(PyExc_ValueError, "Number of classes should be at least 2!");
        return NULL;
    }

    int krnl = 1, nnt_id = 0, kmat_alg = 0;
    printf("====================================\n");
    printf("Running GPC prediction\n");
    printf("Number of classes: %d\n", num_classes);
    int ret = parse_gp_params(
        kernel_type, &krnl, transform, &nnt_id,
        n1, dim, mvtype, &kmat_alg, 
        1, dtype, exact_gp, rank, lfil
    );
    printf("PCG parameters: niter %d, tol %g\n", niter, tol);
    printf("====================================\n");
    if (lfil > 0) npt_s = -(rank+1);  // Mute rank estimation and use AFN
    else npt_s = -rank;               // Mute rank estimation and use Nystrom
    if (ret == 0) return NULL;

    // Determine the return type
    switch (dtype)
    {
        case FP64:
        {
            npy_intp dim[] = {n2};
            npy_intp dim2[] = {num_classes, n2};
            label_prediction    = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_INT32);
            label_value         = (PyArrayObject *) PyArray_SimpleNew(2, dim2, NPY_FLOAT64);
            label_prob          = (PyArrayObject *) PyArray_SimpleNew(2, dim2, NPY_FLOAT64);
            break;
        }
        case FP32:
        {
            npy_intp dim[] = {n2};
            label_prediction    = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_INT32);
            label_value         = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
            label_prob          = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT32);
            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown data type!");
            return NULL;
        }
    }

    switch (dtype)
    {
        case FP64:
        {
            int val_type = VAL_TYPE_DOUBLE;
            double *pyparams_pt = (double *) PyArray_DATA(pyparams);
            double *X_train = (double *) PyArray_DATA(data_train);
            double *X_pred = (double *) PyArray_DATA(data_prediction);
            int *Y_pred = (int *) PyArray_DATA(label_prediction);
            double *Y_value = (double *) PyArray_DATA(label_value);
            double *Y_prob = (double *) PyArray_DATA(label_prob);

            double *param = (double *) malloc(3 * num_classes * sizeof(double));
            for (int i = 0; i < 3 * num_classes; i++)
                param[i] = pyparams_pt[i];

            if (exact_gp)
            {
                exact_gpc_predict(
                    val_type, nnt_id, krnl, num_classes, 
                    nsample, (const void *) &param[0], n1, dim, 
                    (const void *) X_train, n1, training_label_buffer, n2,
                    (const void *) X_pred, n2, Y_pred, (void *) Y_value, (void *) Y_prob
                );
            }
            else
            {
                precond_gpc_predict(
                    val_type, nnt_id, krnl, num_classes,
                    nsample, (const void *) &param[0], n1, dim,
                    (const void *) X_train, n1, training_label_buffer, n2,
                    (const void *) X_pred, n2, npt_s, rank, lfil, niter, 
                    (const void *) &tol, kmat_alg, Y_pred, (void *) Y_value, (void *) Y_prob
                );
            }

            free(param);
            break;
        }
        case FP32:
        {
            int val_type = VAL_TYPE_FLOAT;
            float *pyparams_pt = (float *) PyArray_DATA(pyparams);
            float *X_train = (float *) PyArray_DATA(data_train);
            float *X_pred = (float *) PyArray_DATA(data_prediction);
            int *Y_pred = (int *) PyArray_DATA(label_prediction);
            float *Y_value = (float *) PyArray_DATA(label_value);
            float *Y_prob = (float *) PyArray_DATA(label_prob);
            
            float *param = (float *) malloc(3 * num_classes * sizeof(float));
            for (int i = 0; i < 3 * num_classes; i++)
                param[i] = pyparams_pt[i];

            float tol_f = 1e-5;

            if (exact_gp)
            {
                exact_gpc_predict(
                    val_type, nnt_id, krnl, num_classes, 
                    nsample, (const void *) &param[0], n1, dim, 
                    (const void *) X_train, n1, training_label_buffer, n2,
                    (const void *) X_pred, n2, Y_pred, (void *) Y_value, (void *) Y_prob
                );
            }
            else
            {
                precond_gpc_predict(
                    val_type, nnt_id, krnl, num_classes,
                    nsample, (const void *) &param[0], n1, dim,
                    (const void *) X_train, n1, training_label_buffer, n2,
                    (const void *) X_pred, n2, npt_s, rank, lfil, niter, 
                    (const void *) &tol_f, kmat_alg, Y_pred, (void *) Y_value, (void *) Y_prob
                );
            }

            free(param);
            break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Unknown data type!");
            return NULL;
        }
    }
    
    free(training_label_buffer);
    PyObject* results = PyTuple_Pack(3, label_prediction, label_value, label_prob);
    return results;
}