#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../common.h"
#include "nonneg_transform.h"

template<typename VT>
static void nonneg_transform(const int nnt_id, const VT *val, VT *tval, VT *dval)
{
    VT val_ = *val, tval_ = 0, dval_ = 0;
    if (nnt_id == NNT_SOFTPLUS)
    {
        if (val_ < 20.0)
        {
            tval_ = std::log(1.0 + std::exp(val_));
            dval_ = 1.0 / (1.0 + std::exp(-val_));
        } else {
            tval_ = val_;
            dval_ = 1.0;
        }
    }
    if (nnt_id == NNT_EXP)
    {
        tval_ = std::exp(val_);
        dval_ = tval_;
    }
    if (nnt_id == NNT_SIGMOID)
    {
        tval_ = 1.0 / (std::exp(-val_) + 1.0);
        dval_ = tval_ * (1.0 - tval_);
    }
    *tval = tval_;
    *dval = dval_;
}

// Perform a non-negative transform on a value and return its derivative
void nonneg_transform(const int val_type, const int nnt_id, const void *val, void *tval, void *dval)
{
    if (val_type == VAL_TYPE_DOUBLE) nonneg_transform<double>(nnt_id, (const double *) val, (double *) tval, (double *) dval);
    if (val_type == VAL_TYPE_FLOAT)  nonneg_transform<float> (nnt_id, (const float *)  val, (float *)  tval, (float *)  dval);
}

template<typename VT>
static void inverse_nnt(const int nnt_id, const VT *tval, VT *val)
{
    VT tval_ = *tval, val_ = 0;
    if (nnt_id == NNT_SOFTPLUS)
    {
        if (tval_ < 20.0)
        {
            val_ = std::log(std::exp(tval_) - 1.0);
        } else {
            val_ = tval_;
        }
    }
    if (nnt_id == NNT_EXP)
    {
        val_ = std::log(tval_);
    }
    if (nnt_id == NNT_SIGMOID)
    {
        val_ = std::log(tval_ / (1.0 - tval_));
    }
    *val = val_;
}

// Inverse transform of a non-negative transform
void inverse_nnt(const int val_type, const int nnt_id, const void *tval, void *val)
{
    if (val_type == VAL_TYPE_DOUBLE) inverse_nnt<double>(nnt_id, (const double *) tval, (double *) val);
    if (val_type == VAL_TYPE_FLOAT)  inverse_nnt<float> (nnt_id, (const float *)  tval, (float *)  val);
}
