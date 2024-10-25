#ifndef __H2MAT_PROXY_POINTS_H__
#define __H2MAT_PROXY_POINTS_H__

#include "../common.h"
#include "h2mat_utils.h"
#include "octree.h"

#ifdef __cplusplus
extern "C" {
#endif

// Calculate proxy points for each level of an octree. All nodes at the 
// same level of the octree have the same enclosing box size and use the
// same set of proxy points.
// Input parameters:
//   octree : A constructed octree struct
//   krnl   : Pointer to kernel matrix evaluation function
//   param  : Pointer to kernel matrix evaluation function parameter list
//   reltol : Proxy point selection relative error tolerance
// Output parameters:
//   *lvl_pp : Array of h2m_2dbuf_p, size octree->n_level, i-th element
//             is a pointer to a h2m_2dbuf struct that stores the calculated 
//             proxy points for level i, size pp[i]->nrow * pt_dim, col-major
void h2m_octree_proxy_points(
    octree_p octree, krnl_func krnl, const void *param, 
    const void *reltol, h2m_2dbuf_p **lvl_pp 
);

#ifdef __cplusplus
};
#endif

#endif
