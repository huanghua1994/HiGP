#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "h2mat_typedef.h"

// Free an constructed h2mat struct
void h2mat_free(h2mat_p *h2mat)
{
    h2mat_p h2mat_ = *h2mat;
    if (h2mat_ == NULL) return;
    free(h2mat_->node_n_far);
    free(h2mat_->node_far);
    free(h2mat_->node_n_near);
    free(h2mat_->node_near);
    for (int i = 0; i < h2mat_->n_node; i++)
    {
        h2m_2dbuf_free(&h2mat_->J_coords[i]);
        h2m_2dbuf_free(&h2mat_->J_idxs[i]);
        h2m_2dbuf_free(&h2mat_->V_mats[i]);
    }
    free(h2mat_->J_coords);
    free(h2mat_->J_idxs);
    free(h2mat_->V_mats);
    free(h2mat_);
    *h2mat = NULL;
}
