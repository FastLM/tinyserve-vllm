/*
 * C API for the TinyServe query-aware page kernel.
 * The CUDA entry points live in kernels/tinyserve_kernels.h and are built
 * only when nvcc is available.
 */
#ifndef TINYSERVE_H
#define TINYSERVE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TinyServeKvView {
  const float* keys;     /* [num_pages, page_size, dim] */
  const float* values;   /* [num_pages, page_size, dim] */
  const float* k_min;    /* [num_pages, dim] */
  const float* k_max;    /* [num_pages, dim] */
  const int* page_lens;  /* [num_pages] */
  int num_pages;
  int page_size;
  int dim;
} TinyServeKvView;

/* Returns 0 on success and -1 on an invalid argument. */
int tinyserve_bounding_box_score(const float* query, const float* k_min, const float* k_max, int dim,
                                 float* score);

int tinyserve_select_pages(const float* scores, int num_pages, int k_pages, int n_recent, int* out_ids);

int tinyserve_sparse_decode(const float* query, const TinyServeKvView* kv, int k_pages, int n_recent,
                            float scale, int query_position, float* out, int* selected);

int tinyserve_update_bounds(const float* key, int dim, float* k_min, float* k_max, int reset);

int tinyserve_reduce_bounds(const float* keys, int count, int dim, float* k_min, float* k_max);

#ifdef __cplusplus
}
#endif

#endif
