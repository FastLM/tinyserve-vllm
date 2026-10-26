#include "tinyserve.h"

#include "tinyserve/bounding_box.hpp"
#include "tinyserve/page_cache.hpp"
#include "tinyserve/sparse_attention.hpp"
#include "tinyserve/topk.hpp"

#include <exception>
#include <vector>

namespace {

int fail() { return -1; }

}  // namespace

int tinyserve_bounding_box_score(const float* query, const float* k_min, const float* k_max, int dim,
                                 float* score) {
  if (score == nullptr) return fail();
  try {
    *score = tinyserve::bounding_box_score(query, k_min, k_max, dim);
  } catch (const std::exception&) {
    return fail();
  }
  return 0;
}

int tinyserve_select_pages(const float* scores, int num_pages, int k_pages, int n_recent, int* out_ids) {
  if (out_ids == nullptr || num_pages < 0 || k_pages < 0 || n_recent < 0) return fail();
  if (num_pages > 0 && scores == nullptr) return fail();
  try {
    std::vector<float> score_vec(scores, scores + num_pages);
    const std::vector<int> ids = tinyserve::select_pages(score_vec, k_pages, n_recent);
    for (int i = 0; i < k_pages; ++i) {
      out_ids[i] = i < static_cast<int>(ids.size()) ? ids[static_cast<size_t>(i)] : -1;
    }
  } catch (const std::exception&) {
    return fail();
  }
  return 0;
}

int tinyserve_sparse_decode(const float* query, const TinyServeKvView* kv, int k_pages, int n_recent,
                            float scale, int query_position, float* out, int* selected) {
  if (query == nullptr || kv == nullptr || out == nullptr) return fail();
  tinyserve::KvPageView view;
  view.keys = kv->keys;
  view.values = kv->values;
  view.k_min = kv->k_min;
  view.k_max = kv->k_max;
  view.page_lens = kv->page_lens;
  view.num_pages = kv->num_pages;
  view.page_size = kv->page_size;
  view.dim = kv->dim;
  try {
    tinyserve::sparse_decode(query, view, k_pages, n_recent, scale, query_position, out, selected);
  } catch (const std::exception&) {
    return fail();
  }
  return 0;
}

int tinyserve_update_bounds(const float* key, int dim, float* k_min, float* k_max, int reset) {
  try {
    tinyserve::update_bounds(key, dim, k_min, k_max, reset != 0);
  } catch (const std::exception&) {
    return fail();
  }
  return 0;
}

int tinyserve_reduce_bounds(const float* keys, int count, int dim, float* k_min, float* k_max) {
  try {
    tinyserve::reduce_bounds(keys, count, dim, k_min, k_max);
  } catch (const std::exception&) {
    return fail();
  }
  return 0;
}
