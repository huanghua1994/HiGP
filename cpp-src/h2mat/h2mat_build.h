#ifndef __H2MAT_BUILD_H__
#define __H2MAT_BUILD_H__

#include "../kernels/kernels.h"
#include "octree.h"
#include "h2mat_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build an H2 matrix for K(X, X, l) with a given point partitioning 
// octree and a set of proxy points
// Input parameter:
//   octree : Point partitioning octree
//   pp     : Size octree->n_level, proxy points for each level
//   krnl   : Kernel matrix evaluation function
//   param  : Kernel matrix parameters [dim, l]
//   reltol : H2 compression relative error tolerance
// Output parameter:
//   *h2mat : Constrcuted h2mat struct
// Notes: 
//   1. DO NOT free octree and param before calling h2mat_free()
//   2. octree->val_type specifies the data type of param, reltol, and h2mat->V_mats
void h2mat_build_with_proxy_points(
    octree_p octree, h2m_2dbuf_p *pp, krnl_func krnl,
    void *param, const void *reltol, h2mat_p *h2mat
);

#ifdef __cplusplus
}
#endif

#endif
