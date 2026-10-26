/*
 * TinyServe fused query-aware sparse attention.
 *
 * Algorithm 1 in the paper scores pages, gathers the selected KV, then runs
 * attention. This kernel keeps that math and fuses it:
 *   1. stream page bounds through a tile bitonic top-k (no global score buffer)
 *   2. pin the newest pages
 *   3. double-buffer the selected pages with cp.async on Ampere+
 *   4. online-softmax the page so selected KV is never written back
 *
 * KV is fp16 or int8. Bounds and the query are fp32. Head dim is 32, 64, or
 * 128. Page size is 8, 16, 32, or 64. At most 256 pages are selected and at
 * most 32 recent pages are pinned.
 */

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* kv_kind: 0 = fp16, 1 = int8. Returns 0 on success, 1 on a bad shape, 2 on a CUDA error. */
int tinyserve_fused_sparse_attention(
    const float* q,             /* [B, Hq, D] */
    const void* k,              /* [B, Hkv, P, S, D] */
    const void* v,
    const float* k_min,         /* [B, Hkv, P, D] */
    const float* k_max,
    const int* page_lens,       /* [B, P] */
    const int* num_pages,       /* [B] */
    const float* k_scale,       /* [B, Hkv, P], nullable for fp16 */
    const float* v_scale,
    float* out,                 /* [B, Hq, D] */
    int* selected,              /* [B, Hq, K] */
    int batch,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages,
    int k_pages,
    int n_recent,
    float scale,
    int kv_kind,
    cudaStream_t stream);

/* Incorporate one newly written token into a page's bounds. reset != 0 replaces them. */
int tinyserve_update_bounds(
    const void* key_token,      /* [B, Hkv, D] fp16 */
    float* k_min,               /* [B, Hkv, P, D] */
    float* k_max,
    const int* page_index,      /* [B] */
    const int* reset,           /* [B] */
    int batch,
    int num_kv_heads,
    int max_pages,
    int head_dim,
    cudaStream_t stream);

/* Reduce one full page of fp16 keys into bounds. lengths are the valid token counts. */
int tinyserve_reduce_page_bounds(
    const void* key_page,       /* [B, Hkv, S, D] fp16 */
    float* k_min,               /* [B, Hkv, D] */
    float* k_max,
    const int* lengths,         /* [B] */
    int batch,
    int num_kv_heads,
    int page_size,
    int head_dim,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
