#include "tinyserve_kernels.h"

#include <cuda_fp16.h>
#include <math_constants.h>
#include <stdint.h>

namespace {

constexpr int kTile = 128;
constexpr int kMaxK = 256;
constexpr int kMaxRecent = 32;

__device__ __forceinline__ bool rank_better(float score_a, int id_a, float score_b, int id_b) {
  if (score_a > score_b) return true;
  if (score_a < score_b) return false;
  return id_a < id_b;
}

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
#if __CUDA_ARCH__ >= 800
  unsigned smem_ptr = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_ptr), "l"(gmem_src));
#else
  *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
#endif
}

__device__ __forceinline__ void cp_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_wait_all() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ void cp_wait_one() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}

template <typename KV>
__device__ __forceinline__ float kv_as_float(KV value, float scale);

template <>
__device__ __forceinline__ float kv_as_float<__half>(__half value, float scale) {
  return __half2float(value) * scale;
}

template <>
__device__ __forceinline__ float kv_as_float<int8_t>(int8_t value, float scale) {
  return static_cast<float>(value) * scale;
}

__device__ float block_reduce_sum(float value) {
  __shared__ float warp_sums[4];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float folded = (lane < 4) ? warp_sums[lane] : 0.f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      folded += __shfl_down_sync(0xffffffff, folded, offset);
    }
    if (lane == 0) warp_sums[0] = folded;
  }
  __syncthreads();
  value = warp_sums[0];
  __syncthreads();
  return value;
}

__device__ float block_reduce_max(float value) {
  __shared__ float warp_maxs[4];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  if (lane == 0) warp_maxs[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float folded = (lane < 4) ? warp_maxs[lane] : -CUDART_INF_F;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      folded = fmaxf(folded, __shfl_down_sync(0xffffffff, folded, offset));
    }
    if (lane == 0) warp_maxs[0] = folded;
  }
  __syncthreads();
  value = warp_maxs[0];
  __syncthreads();
  return value;
}

__device__ void bitonic_desc(float* scores, int* ids, int n) {
  const int tid = threadIdx.x;
  for (int span = 2; span <= n; span <<= 1) {
    for (int stride = span >> 1; stride > 0; stride >>= 1) {
      const int partner = tid ^ stride;
      if (partner > tid && tid < n && partner < n) {
        const bool left_better =
            rank_better(scores[tid], ids[tid], scores[partner], ids[partner]);
        const bool swap_pair = ((tid & span) == 0) ? !left_better : left_better;
        if (swap_pair) {
          const float score_tmp = scores[tid];
          scores[tid] = scores[partner];
          scores[partner] = score_tmp;
          const int id_tmp = ids[tid];
          ids[tid] = ids[partner];
          ids[partner] = id_tmp;
        }
      }
      __syncthreads();
    }
  }
}

__device__ void merge_topk_device(
    float* top_scores,
    int* top_ids,
    const float* tile_scores,
    const int* tile_ids,
    float* merged_scores,
    int* merged_ids,
    int k,
    int tile) {
  if (threadIdx.x != 0) return;
  int top_cursor = 0;
  int tile_cursor = 0;
  for (int slot = 0; slot < k; ++slot) {
    bool take_top;
    if (top_cursor >= k) {
      take_top = false;
    } else if (tile_cursor >= tile) {
      take_top = true;
    } else if (rank_better(top_scores[top_cursor], top_ids[top_cursor], tile_scores[tile_cursor],
                           tile_ids[tile_cursor])) {
      take_top = true;
    } else if (rank_better(tile_scores[tile_cursor], tile_ids[tile_cursor], top_scores[top_cursor],
                           top_ids[top_cursor])) {
      take_top = false;
    } else {
      take_top = true;
    }
    if (take_top) {
      merged_scores[slot] = top_scores[top_cursor];
      merged_ids[slot] = top_ids[top_cursor];
      ++top_cursor;
    } else {
      merged_scores[slot] = tile_scores[tile_cursor];
      merged_ids[slot] = tile_ids[tile_cursor];
      ++tile_cursor;
    }
  }
  for (int slot = 0; slot < k; ++slot) {
    top_scores[slot] = merged_scores[slot];
    top_ids[slot] = merged_ids[slot];
  }
}

__device__ void pin_recent_device(
    float* top_scores,
    int* top_ids,
    int* missing,
    int* evictable,
    int k,
    int num_pages,
    int n_recent) {
  if (threadIdx.x != 0) return;
  if (num_pages <= 0 || n_recent <= 0 || k <= 0) return;
  int recent_lo = num_pages - n_recent;
  if (recent_lo < 0) recent_lo = 0;
  int missing_count = 0;
  for (int page = recent_lo; page < num_pages && missing_count < kMaxRecent; ++page) {
    bool found = false;
    for (int slot = 0; slot < k; ++slot) {
      if (top_ids[slot] == page) {
        found = true;
        break;
      }
    }
    if (!found) missing[missing_count++] = page;
  }
  if (missing_count == 0) return;
  int evict_count = 0;
  for (int slot = k - 1; slot >= 0; --slot) {
    const int ident = top_ids[slot];
    const bool is_recent = ident >= recent_lo && ident < num_pages;
    if (!is_recent) evictable[evict_count++] = slot;
  }
  if (missing_count > evict_count) {
    int start = num_pages - k;
    if (start < 0) start = 0;
    const int keep = num_pages - start;
    for (int slot = 0; slot < k; ++slot) {
      if (slot < keep) {
        top_ids[slot] = start + slot;
        top_scores[slot] = 1.0e30f;
      } else {
        top_ids[slot] = -1;
        top_scores[slot] = -CUDART_INF_F;
      }
    }
    return;
  }
  for (int index = 0; index < missing_count; ++index) {
    const int slot = evictable[index];
    top_ids[slot] = missing[index];
    top_scores[slot] = 1.0e30f;
  }
}

template <int kHeadDim>
__device__ float bbox_score(const float* query, const float* k_min, const float* k_max) {
  float acc = 0.f;
#pragma unroll
  for (int dim = 0; dim < kHeadDim; dim += 4) {
    const float4 lower = *reinterpret_cast<const float4*>(k_min + dim);
    const float4 upper = *reinterpret_cast<const float4*>(k_max + dim);
    const float q0 = query[dim];
    const float q1 = query[dim + 1];
    const float q2 = query[dim + 2];
    const float q3 = query[dim + 3];
    acc += q0 * ((q0 >= 0.f) ? upper.x : lower.x);
    acc += q1 * ((q1 >= 0.f) ? upper.y : lower.y);
    acc += q2 * ((q2 >= 0.f) ? upper.z : lower.z);
    acc += q3 * ((q3 >= 0.f) ? upper.w : lower.w);
  }
  return acc;
}

template <typename KV, int kPageSize, int kHeadDim>
__device__ void load_tiles(KV* dst_k, KV* dst_v, const KV* src_k, const KV* src_v) {
  constexpr int kVec = 16 / static_cast<int>(sizeof(KV));
  constexpr int kNumVec = (kPageSize * kHeadDim) / kVec;
  for (int index = threadIdx.x; index < kNumVec; index += blockDim.x) {
    cp_async_16(dst_k + index * kVec, src_k + index * kVec);
    cp_async_16(dst_v + index * kVec, src_v + index * kVec);
  }
}

template <typename KV, int kPageSize, int kHeadDim>
__device__ const KV* page_pointer(
    const KV* base, int batch, int kv_head, int page, int num_kv_heads, int max_pages) {
  constexpr int kElems = kPageSize * kHeadDim;
  const size_t index =
      (static_cast<size_t>(batch) * num_kv_heads + kv_head) * max_pages + page;
  return base + index * kElems;
}

__device__ float page_scale_at(
    const float* scale, int batch, int kv_head, int page, int num_kv_heads, int max_pages) {
  if (scale == nullptr) return 1.f;
  return scale[(static_cast<size_t>(batch) * num_kv_heads + kv_head) * max_pages + page];
}

template <typename KV, int kPageSize, int kHeadDim, bool kDouble>
__device__ KV* smem_tile(char* base, int buffer, int which) {
  constexpr int kElems = kPageSize * kHeadDim;
  const int slot = kDouble ? (buffer * 2 + which) : which;
  return reinterpret_cast<KV*>(base) + static_cast<size_t>(slot) * kElems;
}

template <typename KV, int kHeadDim, int kPageSize>
__device__ void attend_page(
    const float* query,
    const KV* key_tile,
    const KV* value_tile,
    float* dots,
    int page_len,
    float key_scale,
    float value_scale,
    float attn_scale,
    float& running_max,
    float& running_sum,
    float& accumulator) {
  if (page_len <= 0) return;
  const int tid = threadIdx.x;
  for (int token = 0; token < page_len; ++token) {
    float partial = 0.f;
    for (int dim = tid; dim < kHeadDim; dim += blockDim.x) {
      partial += query[dim] * kv_as_float(key_tile[token * kHeadDim + dim], key_scale);
    }
    const float dot = block_reduce_sum(partial) * attn_scale;
    if (tid == 0) dots[token] = dot;
    __syncthreads();
  }

  float local_max = -CUDART_INF_F;
  for (int token = tid; token < page_len; token += blockDim.x) {
    local_max = fmaxf(local_max, dots[token]);
  }
  const float tile_max = block_reduce_max(local_max);
  const float next_max = fmaxf(running_max, tile_max);
  const float alpha = (running_max == -CUDART_INF_F) ? 0.f : expf(running_max - next_max);
  float local_sum = 0.f;
  for (int token = tid; token < page_len; token += blockDim.x) {
    local_sum += expf(dots[token] - next_max);
  }
  const float sum_exp = block_reduce_sum(local_sum);
  if (tid < kHeadDim) {
    float weighted = 0.f;
    for (int token = 0; token < page_len; ++token) {
      const float prob = expf(dots[token] - next_max);
      weighted += prob * kv_as_float(value_tile[token * kHeadDim + tid], value_scale);
    }
    accumulator = accumulator * alpha + weighted;
  }
  running_sum = running_sum * alpha + sum_exp;
  running_max = next_max;
}

template <typename KV, int kHeadDim, int kPageSize, bool kDouble>
__global__ void fused_query_aware_kernel(
    const float* __restrict__ q,
    const KV* __restrict__ k,
    const KV* __restrict__ v,
    const float* __restrict__ k_min,
    const float* __restrict__ k_max,
    const int* __restrict__ page_lens,
    const int* __restrict__ num_pages,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    float* __restrict__ out,
    int* __restrict__ selected,
    int num_q_heads,
    int num_kv_heads,
    int max_pages,
    int k_pages,
    int n_recent,
    float attn_scale) {
  const int batch = blockIdx.x;
  const int q_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int group = num_q_heads / num_kv_heads;
  const int kv_head = q_head / group;

  __shared__ float top_scores[kMaxK];
  __shared__ int top_ids[kMaxK];
  __shared__ float tile_scores[kTile];
  __shared__ int tile_ids[kTile];
  __shared__ float merged_scores[kMaxK];
  __shared__ int merged_ids[kMaxK];
  __shared__ float query_s[kHeadDim];
  __shared__ float dots[kPageSize];
  __shared__ int valid_ids[kMaxK];
  __shared__ int nvalid_s;
  __shared__ float scale_k[2];
  __shared__ float scale_v[2];
  __shared__ int missing[kMaxRecent];
  __shared__ int evictable[kMaxK];
  extern __shared__ __align__(16) char dyn_smem[];

  const float* q_ptr = q + (static_cast<size_t>(batch) * num_q_heads + q_head) * kHeadDim;
  for (int dim = tid; dim < kHeadDim; dim += blockDim.x) query_s[dim] = q_ptr[dim];
  for (int slot = tid; slot < k_pages; slot += blockDim.x) {
    top_scores[slot] = -CUDART_INF_F;
    top_ids[slot] = -1;
  }
  __syncthreads();

  int npages = num_pages[batch];
  if (npages > max_pages) npages = max_pages;
  if (npages < 0) npages = 0;

  for (int base = 0; base < npages; base += kTile) {
    const int page = base + tid;
    float score = -CUDART_INF_F;
    int ident = -1;
    if (page < npages) {
      const int len = page_lens[static_cast<size_t>(batch) * max_pages + page];
      if (len > 0) {
        const size_t bound_index =
            ((static_cast<size_t>(batch) * num_kv_heads + kv_head) * max_pages + page) * kHeadDim;
        score = bbox_score<kHeadDim>(query_s, k_min + bound_index, k_max + bound_index);
        ident = page;
      }
    }
    tile_scores[tid] = score;
    tile_ids[tid] = ident;
    __syncthreads();
    bitonic_desc(tile_scores, tile_ids, kTile);
    __syncthreads();
    merge_topk_device(
        top_scores, top_ids, tile_scores, tile_ids, merged_scores, merged_ids, k_pages, kTile);
    __syncthreads();
  }

  pin_recent_device(top_scores, top_ids, missing, evictable, k_pages, npages, n_recent);
  __syncthreads();

  if (tid == 0) {
    int count = 0;
    for (int slot = 0; slot < k_pages; ++slot) {
      const int ident = top_ids[slot];
      if (ident < 0 || ident >= npages) continue;
      bool seen = false;
      for (int prev = 0; prev < count; ++prev) {
        if (valid_ids[prev] == ident) {
          seen = true;
          break;
        }
      }
      if (!seen) valid_ids[count++] = ident;
    }
    nvalid_s = count;
  }
  __syncthreads();
  const int nvalid = nvalid_s;

  for (int slot = tid; slot < k_pages; slot += blockDim.x) {
    selected[(static_cast<size_t>(batch) * num_q_heads + q_head) * k_pages + slot] = top_ids[slot];
  }

  float running_max = -CUDART_INF_F;
  float running_sum = 0.f;
  float accumulator = 0.f;

  if (nvalid > 0) {
    if constexpr (kDouble) {
      const int first = valid_ids[0];
      load_tiles<KV, kPageSize, kHeadDim>(
          smem_tile<KV, kPageSize, kHeadDim, true>(dyn_smem, 0, 0),
          smem_tile<KV, kPageSize, kHeadDim, true>(dyn_smem, 0, 1),
          page_pointer<KV, kPageSize, kHeadDim>(k, batch, kv_head, first, num_kv_heads, max_pages),
          page_pointer<KV, kPageSize, kHeadDim>(v, batch, kv_head, first, num_kv_heads, max_pages));
      if (tid == 0) {
        scale_k[0] = page_scale_at(k_scale, batch, kv_head, first, num_kv_heads, max_pages);
        scale_v[0] = page_scale_at(v_scale, batch, kv_head, first, num_kv_heads, max_pages);
      }
      cp_commit();
      for (int index = 0; index < nvalid; ++index) {
        const int current = index & 1;
        const int next = current ^ 1;
        if (index + 1 < nvalid) {
          const int upcoming = valid_ids[index + 1];
          load_tiles<KV, kPageSize, kHeadDim>(
              smem_tile<KV, kPageSize, kHeadDim, true>(dyn_smem, next, 0),
              smem_tile<KV, kPageSize, kHeadDim, true>(dyn_smem, next, 1),
              page_pointer<KV, kPageSize, kHeadDim>(k, batch, kv_head, upcoming, num_kv_heads, max_pages),
              page_pointer<KV, kPageSize, kHeadDim>(v, batch, kv_head, upcoming, num_kv_heads, max_pages));
          if (tid == 0) {
            scale_k[next] = page_scale_at(k_scale, batch, kv_head, upcoming, num_kv_heads, max_pages);
            scale_v[next] = page_scale_at(v_scale, batch, kv_head, upcoming, num_kv_heads, max_pages);
          }
          cp_commit();
          cp_wait_one();
        } else {
          cp_wait_all();
        }
        __syncthreads();
        const int page = valid_ids[index];
        int len = page_lens[static_cast<size_t>(batch) * max_pages + page];
        if (len > kPageSize) len = kPageSize;
        if (len < 0) len = 0;
        attend_page<KV, kHeadDim, kPageSize>(
            query_s,
            smem_tile<KV, kPageSize, kHeadDim, true>(dyn_smem, current, 0),
            smem_tile<KV, kPageSize, kHeadDim, true>(dyn_smem, current, 1),
            dots,
            len,
            scale_k[current],
            scale_v[current],
            attn_scale,
            running_max,
            running_sum,
            accumulator);
        __syncthreads();
      }
    } else {
      for (int index = 0; index < nvalid; ++index) {
        const int page = valid_ids[index];
        load_tiles<KV, kPageSize, kHeadDim>(
            smem_tile<KV, kPageSize, kHeadDim, false>(dyn_smem, 0, 0),
            smem_tile<KV, kPageSize, kHeadDim, false>(dyn_smem, 0, 1),
            page_pointer<KV, kPageSize, kHeadDim>(k, batch, kv_head, page, num_kv_heads, max_pages),
            page_pointer<KV, kPageSize, kHeadDim>(v, batch, kv_head, page, num_kv_heads, max_pages));
        if (tid == 0) {
          scale_k[0] = page_scale_at(k_scale, batch, kv_head, page, num_kv_heads, max_pages);
          scale_v[0] = page_scale_at(v_scale, batch, kv_head, page, num_kv_heads, max_pages);
        }
        cp_commit();
        cp_wait_all();
        __syncthreads();
        int len = page_lens[static_cast<size_t>(batch) * max_pages + page];
        if (len > kPageSize) len = kPageSize;
        if (len < 0) len = 0;
        attend_page<KV, kHeadDim, kPageSize>(
            query_s,
            smem_tile<KV, kPageSize, kHeadDim, false>(dyn_smem, 0, 0),
            smem_tile<KV, kPageSize, kHeadDim, false>(dyn_smem, 0, 1),
            dots,
            len,
            scale_k[0],
            scale_v[0],
            attn_scale,
            running_max,
            running_sum,
            accumulator);
        __syncthreads();
      }
    }
  }

  if (tid < kHeadDim) {
    const bool finite = running_sum > 0.f && isfinite(running_sum);
    out[(static_cast<size_t>(batch) * num_q_heads + q_head) * kHeadDim + tid] =
        finite ? accumulator / running_sum : 0.f;
  }
}

template <typename KV, int kHeadDim, int kPageSize>
int launch_fused(
    const float* q,
    const KV* k,
    const KV* v,
    const float* k_min,
    const float* k_max,
    const int* page_lens,
    const int* num_pages,
    const float* k_scale,
    const float* v_scale,
    float* out,
    int* selected,
    int batch,
    int num_q_heads,
    int num_kv_heads,
    int max_pages,
    int k_pages,
    int n_recent,
    float scale,
    cudaStream_t stream) {
  constexpr int kBytes = kPageSize * kHeadDim * static_cast<int>(sizeof(KV));
  constexpr bool kDouble = (4 * kBytes) <= 32 * 1024;
  constexpr int kDynamic = (kDouble ? 4 : 2) * kBytes;
  auto kernel = fused_query_aware_kernel<KV, kHeadDim, kPageSize, kDouble>;
  kernel<<<dim3(batch, num_q_heads), kTile, kDynamic, stream>>>(
      q, k, v, k_min, k_max, page_lens, num_pages, k_scale, v_scale, out, selected, num_q_heads,
      num_kv_heads, max_pages, k_pages, n_recent, scale);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

template <typename KV, int kHeadDim>
int dispatch_page(
    const float* q,
    const void* k,
    const void* v,
    const float* k_min,
    const float* k_max,
    const int* page_lens,
    const int* num_pages,
    const float* k_scale,
    const float* v_scale,
    float* out,
    int* selected,
    int batch,
    int num_q_heads,
    int num_kv_heads,
    int page_size,
    int max_pages,
    int k_pages,
    int n_recent,
    float scale,
    cudaStream_t stream) {
  const KV* key = static_cast<const KV*>(k);
  const KV* value = static_cast<const KV*>(v);
  switch (page_size) {
    case 8:
      return launch_fused<KV, kHeadDim, 8>(
          q, key, value, k_min, k_max, page_lens, num_pages, k_scale, v_scale, out, selected, batch,
          num_q_heads, num_kv_heads, max_pages, k_pages, n_recent, scale, stream);
    case 16:
      return launch_fused<KV, kHeadDim, 16>(
          q, key, value, k_min, k_max, page_lens, num_pages, k_scale, v_scale, out, selected, batch,
          num_q_heads, num_kv_heads, max_pages, k_pages, n_recent, scale, stream);
    case 32:
      return launch_fused<KV, kHeadDim, 32>(
          q, key, value, k_min, k_max, page_lens, num_pages, k_scale, v_scale, out, selected, batch,
          num_q_heads, num_kv_heads, max_pages, k_pages, n_recent, scale, stream);
    case 64:
      return launch_fused<KV, kHeadDim, 64>(
          q, key, value, k_min, k_max, page_lens, num_pages, k_scale, v_scale, out, selected, batch,
          num_q_heads, num_kv_heads, max_pages, k_pages, n_recent, scale, stream);
    default:
      return 1;
  }
}

__global__ void update_bounds_kernel(
    const __half* key_token,
    float* k_min,
    float* k_max,
    const int* page_index,
    const int* reset,
    int num_kv_heads,
    int max_pages,
    int head_dim) {
  const int batch = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int page = page_index[batch];
  const bool replace = reset[batch] != 0;
  const __half* source = key_token + (static_cast<size_t>(batch) * num_kv_heads + kv_head) * head_dim;
  float* lower =
      k_min + ((static_cast<size_t>(batch) * num_kv_heads + kv_head) * max_pages + page) * head_dim;
  float* upper =
      k_max + ((static_cast<size_t>(batch) * num_kv_heads + kv_head) * max_pages + page) * head_dim;
  for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    const float value = __half2float(source[dim]);
    if (replace) {
      lower[dim] = value;
      upper[dim] = value;
    } else {
      lower[dim] = fminf(lower[dim], value);
      upper[dim] = fmaxf(upper[dim], value);
    }
  }
}

__global__ void reduce_page_bounds_kernel(
    const __half* key_page,
    float* k_min,
    float* k_max,
    const int* lengths,
    int num_kv_heads,
    int page_size,
    int head_dim) {
  const int batch = blockIdx.x;
  const int kv_head = blockIdx.y;
  int len = lengths[batch];
  if (len > page_size) len = page_size;
  if (len < 0) len = 0;
  const __half* source =
      key_page + (static_cast<size_t>(batch) * num_kv_heads + kv_head) * page_size * head_dim;
  float* lower = k_min + (static_cast<size_t>(batch) * num_kv_heads + kv_head) * head_dim;
  float* upper = k_max + (static_cast<size_t>(batch) * num_kv_heads + kv_head) * head_dim;
  for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    if (len == 0) {
      lower[dim] = 0.f;
      upper[dim] = 0.f;
      continue;
    }
    float lo = __half2float(source[dim]);
    float hi = lo;
    for (int token = 1; token < len; ++token) {
      const float value = __half2float(source[token * head_dim + dim]);
      lo = fminf(lo, value);
      hi = fmaxf(hi, value);
    }
    lower[dim] = lo;
    upper[dim] = hi;
  }
}

}  // namespace

extern "C" int tinyserve_fused_sparse_attention(
    const float* q,
    const void* k,
    const void* v,
    const float* k_min,
    const float* k_max,
    const int* page_lens,
    const int* num_pages,
    const float* k_scale,
    const float* v_scale,
    float* out,
    int* selected,
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
    cudaStream_t stream) {
  if (batch < 0 || num_q_heads <= 0 || num_kv_heads <= 0 || max_pages <= 0) return 1;
  if (num_q_heads % num_kv_heads != 0) return 1;
  if (k_pages <= 0 || k_pages > kMaxK || n_recent < 0 || n_recent > kMaxRecent) return 1;
  if (q == nullptr || k == nullptr || v == nullptr || k_min == nullptr || k_max == nullptr) return 1;
  if (page_lens == nullptr || num_pages == nullptr || out == nullptr || selected == nullptr) return 1;
  if (kv_kind == 1 && (k_scale == nullptr || v_scale == nullptr)) return 1;
  if (batch == 0) return 0;

  if (kv_kind == 0) {
    switch (head_dim) {
      case 32:
        return dispatch_page<__half, 32>(
            q, k, v, k_min, k_max, page_lens, num_pages, k_scale, v_scale, out, selected, batch,
            num_q_heads, num_kv_heads, page_size, max_pages, k_pages, n_recent, scale, stream);
      case 64:
        return dispatch_page<__half, 64>(
            q, k, v, k_min, k_max, page_lens, num_pages, k_scale, v_scale, out, selected, batch,
            num_q_heads, num_kv_heads, page_size, max_pages, k_pages, n_recent, scale, stream);
      case 128:
        return dispatch_page<__half, 128>(
            q, k, v, k_min, k_max, page_lens, num_pages, k_scale, v_scale, out, selected, batch,
            num_q_heads, num_kv_heads, page_size, max_pages, k_pages, n_recent, scale, stream);
      default:
        return 1;
    }
  }
  if (kv_kind == 1) {
    switch (head_dim) {
      case 32:
        return dispatch_page<int8_t, 32>(
            q, k, v, k_min, k_max, page_lens, num_pages, k_scale, v_scale, out, selected, batch,
            num_q_heads, num_kv_heads, page_size, max_pages, k_pages, n_recent, scale, stream);
      case 64:
        return dispatch_page<int8_t, 64>(
            q, k, v, k_min, k_max, page_lens, num_pages, k_scale, v_scale, out, selected, batch,
            num_q_heads, num_kv_heads, page_size, max_pages, k_pages, n_recent, scale, stream);
      case 128:
        return dispatch_page<int8_t, 128>(
            q, k, v, k_min, k_max, page_lens, num_pages, k_scale, v_scale, out, selected, batch,
            num_q_heads, num_kv_heads, page_size, max_pages, k_pages, n_recent, scale, stream);
      default:
        return 1;
    }
  }
  return 1;
}

extern "C" int tinyserve_update_bounds(
    const void* key_token,
    float* k_min,
    float* k_max,
    const int* page_index,
    const int* reset,
    int batch,
    int num_kv_heads,
    int max_pages,
    int head_dim,
    cudaStream_t stream) {
  if (batch < 0 || num_kv_heads <= 0 || max_pages <= 0 || head_dim <= 0) return 1;
  if (key_token == nullptr || k_min == nullptr || k_max == nullptr || page_index == nullptr ||
      reset == nullptr) {
    return 1;
  }
  if (batch == 0) return 0;
  update_bounds_kernel<<<dim3(batch, num_kv_heads), 128, 0, stream>>>(
      static_cast<const __half*>(key_token), k_min, k_max, page_index, reset, num_kv_heads,
      max_pages, head_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int tinyserve_reduce_page_bounds(
    const void* key_page,
    float* k_min,
    float* k_max,
    const int* lengths,
    int batch,
    int num_kv_heads,
    int page_size,
    int head_dim,
    cudaStream_t stream) {
  if (batch < 0 || num_kv_heads <= 0 || page_size <= 0 || head_dim <= 0) return 1;
  if (key_page == nullptr || k_min == nullptr || k_max == nullptr || lengths == nullptr) return 1;
  if (batch == 0) return 0;
  reduce_page_bounds_kernel<<<dim3(batch, num_kv_heads), 128, 0, stream>>>(
      static_cast<const __half*>(key_page), k_min, k_max, lengths, num_kv_heads, page_size, head_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
