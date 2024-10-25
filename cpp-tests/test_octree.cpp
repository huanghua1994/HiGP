#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "common.h"
#include "h2mat/h2mat.h"

template <typename VT>
int test_function(const int argc, const char **argv)
{
    const int is_float  = std::is_same<VT, float>::value;
    const int is_double = std::is_same<VT, double>::value;

    if (is_double) printf("========== Test double precision ==========\n");
    if (is_float)  printf("========== Test single precision ==========\n");

    val_type_t val_type = is_double ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;

    int npt = atoi(argv[1]);
    int pt_dim = atoi(argv[2]);
    VT *coord = (VT *) malloc(sizeof(VT) * npt * pt_dim);
    if (argc >= 4)
    {
        FILE *inf = fopen(argv[3], "rb");
        fread(coord, sizeof(VT), npt * pt_dim, inf);
        fclose(inf);
    } else {
        for (int i = 0; i < npt * pt_dim; i++) coord[i] = (VT) rand() / RAND_MAX;
    }

    int leaf_nmax = 400;
    VT leaf_emax = 0;
    octree_p octree = NULL;
    octree_build(npt, pt_dim, val_type, coord, leaf_nmax, (const void *) &leaf_emax, &octree);

    printf("Octree has %d nodes, %d levels\n", octree->n_node, octree->n_level);
    for (int i = 0; i < octree->n_level; i++)
    {
        printf("Level %d has %d nodes: ", i, octree->lvl_nnode[i]);
        for (int j = octree->ln_displs[i]; j < octree->ln_displs[i + 1]; j++)
            printf("%d ", octree->lvl_nodes[j] + 1);
        printf("\n");
    }
    printf(" node | level | # children | parent | point cluster | enbox corner | enbox size | points outside enbox\n");
    int num_out_pt = 0;
    for (int i = 0; i < octree->n_node; i++)
    {
        int clu_s = octree->pt_cluster[i * 2];
        int clu_e = octree->pt_cluster[i * 2 + 1];
        printf("%4d | %2d | %2d | ", i, octree->node_lvl[i], octree->n_children[i]);
        printf("%4d | [%6d, %6d] (%6d) | ", octree->parent[i], clu_s, clu_e, clu_e - clu_s + 1);
        printf("(");
        VT *enbox_i = ((VT *) octree->enbox) + i * 2 * pt_dim;
        for (int j = 0; j < pt_dim-1; j++) printf("%6.3f ", enbox_i[j]);
        printf("%6.3f) | (", enbox_i[pt_dim-1]);
        for (int j = 0; j < pt_dim-1; j++) printf("%6.3f ", enbox_i[j + pt_dim]);
        printf("%6.3f) | ", enbox_i[2 * pt_dim-1]);

        int pt_in_enbox = 0;
        int clu_size = clu_e - clu_s + 1;
        VT x_j[3];
        for (int j = clu_s; j <= clu_e; j++)
        {
            for (int k = 0; k < pt_dim; k++) x_j[k] = coord[octree->bwd_perm[j] + k * npt];
            if (h2m_is_point_in_enbox(val_type, pt_dim, x_j, enbox_i)) pt_in_enbox++;
        }
        printf("%d / %d\n", pt_in_enbox - clu_size, clu_size);
        num_out_pt += pt_in_enbox - clu_size;
    }

    int test_passed = (num_out_pt == 0) ? 1 : 0;
    if (test_passed) printf("Test passed\n\n");
    else printf("Test failed\n\n");

    octree_free(&octree);
    free(coord);
    return test_passed;
}

int main(const int argc, const char **argv)
{
    if (argc < 3)
    {
        printf("Usage: %s n_point pt_dim coord_bin(optional)\n", argv[0]);
        printf("  n_point   : Number of points\n");
        printf("  pt_dim    : Point dimension\n");
        printf("  coord_bin : Binary file containing point coordinates, col-major, size n_point * pt_dim\n");
        return 255;
    }

    int fp32_passed = test_function<float>(argc, argv);
    int fp64_passed = test_function<double>(argc, argv);
    int test_passed = fp32_passed && fp64_passed;
    printf("Are all tests passed? %s\n\n", test_passed ? "YES" : "NO");
    
    return (1 - test_passed);
}
