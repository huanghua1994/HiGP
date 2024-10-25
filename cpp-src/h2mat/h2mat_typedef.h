#ifndef __H2MAT_TYPEDEF_H__
#define __H2MAT_TYPEDEF_H__

#include "octree.h"
#include "h2mat_utils.h"
#include "../kernels/kernels.h"

struct h2mat
{
    octree_p octree;            // Point partitioning octree
    int n_node;                 // Number of nodes in the octree
    int min_far_lvl;            // Minimum level of far pairs
    int *node_n_far;            // Size octree->n_node, number of near pairs for each node
    int *node_far;              // Size octree->n_node * octree->n_node, each row stores the near pairs for a node
    int *node_n_near;           // Size octree->n_node, number of far pairs for each node
    int *node_near;             // Size octree->n_node * octree->n_node, each row stores the far pairs for a node
    krnl_func krnl;             // Kernel matrix evaluate function
    void *param;                // Kernel matrix parameters [dim, l]
    h2m_2dbuf_p *J_coords;      // Size octree->n_node, col-major, each row is a skeleton point indices of a node
    h2m_2dbuf_p *J_idxs;        // Size octree->n_node, J_idxs[i] are the indices of J_coords[i] in octree->px
    h2m_2dbuf_p *V_mats;        // Size octree->n_node, row basis matrices stored in col-major of each node
};
typedef struct h2mat  h2mat_s;
typedef struct h2mat* h2mat_p;

#ifdef __cplusplus
extern "C" {
#endif

// h2mat_build() is in h2mate_build.h

// Free an constructed h2mat struct
void h2mat_free(h2mat_p *h2mat);

#ifdef __cplusplus
}
#endif

#endif
