#ifndef __NONNEG_TRANSFORM_H__
#define __NONNEG_TRANSFORM_H__

typedef enum nonneg_transform
{
    NNT_SOFTPLUS = 0,
    NNT_EXP      = 1,
    NNT_SIGMOID  = 2
} nonneg_transform_t;

#ifdef __cplusplus
extern "C" {
#endif

// Perform a non-negative transform on a value and return its derivative
// Input parameters:
//   val_type : Data type of val, tval, and dval, 0 for double, 1 for float
//   nnt_id   : Non-negative transform ID, 0 for softplus, 1 for exp, 2 for sigmoid
//   val      : Input value
// Output parameters:
//   tval : Transformed value
//   dval : Derivative of the transform w.r.t. val
void nonneg_transform(const int val_type, const int nnt_id, const void *val, void *tval, void *dval);

// Inverse transform of a non-negative transform
// Input and output parameters are the reverse of nonneg_transform()
void inverse_nnt(const int val_type, const int nnt_id, const void *tval, void *val);

#ifdef __cplusplus
}
#endif

#endif
