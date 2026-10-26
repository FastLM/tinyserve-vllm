#pragma once

namespace tinyserve {

struct KvPageView {
  const float* keys;      // [num_pages, page_size, dim]
  const float* values;    // [num_pages, page_size, dim]
  const float* k_min;     // [num_pages, dim]
  const float* k_max;     // [num_pages, dim]
  const int* page_lens;   // [num_pages]
  int num_pages;
  int page_size;
  int dim;
};

// Full attention of one query over a contiguous token list [tokens, dim].
void dense_decode(const float* query, const float* keys, const float* values, int tokens, int dim,
                  float scale, float* out);

// CPU form of the fused kernel: score pages, pin the newest causal pages,
// then softmax over the selected tokens. selected may be null. When it is
// not, it receives k_pages ids, padded with -1.
void sparse_decode(const float* query, const KvPageView& kv, int k_pages, int n_recent, float scale,
                   int query_position, float* out, int* selected);

}  // namespace tinyserve
