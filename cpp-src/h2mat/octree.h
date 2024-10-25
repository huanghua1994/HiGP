#ifndef __OCTREE_H__
#define __OCTREE_H__

// Binary tree, quadtree, or octree
struct octree
{
    int  npt, pt_dim;       // Number of points and point dimension (<= 3)
    int  n_level;           // Number of levels, root node (box) at level 0
    int  n_node;            // Number of nodes, root node has index n_node-1
    int  val_type;          // Data type of px and enbox, 0 for double, 1 for float
    int  *parent;           // Size n_node, parent node index of each node
    int  *children;         // Size n_node * (2^pt_dim), row-major, each row contains children indices of each node
    int  *n_children;       // Size n_node, number of children of each node
    int  *node_lvl;         // Size n_node, level of each node, root node at level 0
    int  *node_npt;         // Size n_node, number of points in each node
    int  *lvl_nnode;        // Size n_level, number of nodes in each level
    int  *lvl_nodes;        // Size n_node, indices of nodes in each level
    int  *ln_displs;        // Size n_level+1, i-th level's node indices are lvl_nodes[ln_displs[i] : ln_displs[i+1]-1]
    int  *pt_cluster;       // Size n_node * 2, row-major, start and end (included) indices of permuted points in each node
    int  *fwd_perm;         // Size npt, the i-th original point is the fwd_perm[i]-th permuted points in the octree
    int  *bwd_perm;         // Size npt, the i-th permuted points in the octree is the bwd_perm[i]-th original point
    void *px;               // Size npt * pt_dim, col-major, coordinates of permuted points
    void *enbox;            // Size n_node * (2 * pt_dim), row-major, each row contains the enclosing box of a node, 
                            // the first pt_dim values are the lower corner, the last pt_dim values are the sizes of the enclosing box
};
typedef struct octree  octree_s;
typedef struct octree *octree_p;

#ifdef __cplusplus
extern "C" {
#endif

// Build an octree for a given point set
// Input parameters:
//   npt, pt_dim : Number of points and point dimension (<= 3)
//   val_type    : Data type of coord, 0 for double, 1 for float
//   coord       : Size npt * pt_dim, col-major, each row is a point coordinate
//   leaf_nmax   : Maximum number of points in a leaf node
//   leaf_emax   : Maximum size of the enclosing box of a leaf node
// Output parameter:
//   *octree : Pointer to a constructed octree struct
void octree_build(
    const int npt, const int pt_dim, const int val_type, const void *coord, 
    const int leaf_nmax, const void *leaf_emax, octree_p *octree
);

// Free an octree struct
void octree_free(octree_p *octree);

#ifdef __cplusplus
}
#endif

#endif  // "#ifndef __OCTREE_H__"
