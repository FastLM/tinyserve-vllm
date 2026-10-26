#include "tinyserve/sparse_attention.hpp"

#include "tinyserve/bounding_box.hpp"
#include "tinyserve/topk.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tinyserve {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

void softmax_weighted_sum(const float* query, const std::vector<const float*>& keys,
                          const std::vector<const float*>& values, int dim, float scale, float* out) {
  const int n = static_cast<int>(keys.size());
  for (int d = 0; d < dim; ++d) out[d] = 0.f;
  if (n == 0) return;

  std::vector<float> logits(static_cast<size_t>(n));
  float max_logit = kNegInf;
  for (int i = 0; i < n; ++i) {
    float dot = 0.f;
    const float* key = keys[static_cast<size_t>(i)];
    for (int d = 0; d < dim; ++d) dot += query[d] * key[d];
    logits[static_cast<size_t>(i)] = dot * scale;
    if (logits[static_cast<size_t>(i)] > max_logit) max_logit = logits[static_cast<size_t>(i)];
  }
  float denom = 0.f;
  for (float logit : logits) denom += std::exp(logit - max_logit);
  if (!(denom > 0.f)) return;
  for (int i = 0; i < n; ++i) {
    const float weight = std::exp(logits[static_cast<size_t>(i)] - max_logit) / denom;
    const float* value = values[static_cast<size_t>(i)];
    for (int d = 0; d < dim; ++d) out[d] += weight * value[d];
  }
}

}  // namespace

void dense_decode(const float* query, const float* keys, const float* values, int tokens, int dim,
                  float scale, float* out) {
  if (query == nullptr || keys == nullptr || values == nullptr || out == nullptr || dim <= 0 || tokens < 0) {
    throw std::invalid_argument("dense_decode received an invalid buffer");
  }
  std::vector<const float*> key_rows;
  std::vector<const float*> value_rows;
  key_rows.reserve(static_cast<size_t>(tokens));
  value_rows.reserve(static_cast<size_t>(tokens));
  for (int t = 0; t < tokens; ++t) {
    key_rows.push_back(keys + static_cast<size_t>(t) * dim);
    value_rows.push_back(values + static_cast<size_t>(t) * dim);
  }
  softmax_weighted_sum(query, key_rows, value_rows, dim, scale, out);
}

void sparse_decode(const float* query, const KvPageView& kv, int k_pages, int n_recent, float scale,
                   int query_position, float* out, int* selected) {
  if (query == nullptr || out == nullptr || kv.dim <= 0 || kv.page_size <= 0 || kv.num_pages < 0) {
    throw std::invalid_argument("sparse_decode received an invalid buffer");
  }
  if (kv.num_pages > 0 &&
      (kv.keys == nullptr || kv.values == nullptr || kv.k_min == nullptr || kv.k_max == nullptr ||
       kv.page_lens == nullptr)) {
    throw std::invalid_argument("sparse_decode is missing a KV buffer");
  }

  const int dim = kv.dim;
  const int page_size = kv.page_size;
  for (int d = 0; d < dim; ++d) out[d] = 0.f;
  if (kv.num_pages == 0 || query_position < 0 || k_pages <= 0) {
    if (selected != nullptr) {
      for (int i = 0; i < k_pages; ++i) selected[i] = -1;
    }
    return;
  }

  int query_page = query_position / page_size;
  if (query_page >= kv.num_pages) query_page = kv.num_pages - 1;
  const int visible = query_page + 1;

  std::vector<float> scores(static_cast<size_t>(visible), kNegInf);
  for (int page = 0; page < visible; ++page) {
    const int len = kv.page_lens[page];
    if (len <= 0) continue;
    if (page * page_size > query_position) continue;
    scores[static_cast<size_t>(page)] = bounding_box_score(
        query, kv.k_min + static_cast<size_t>(page) * dim, kv.k_max + static_cast<size_t>(page) * dim, dim);
  }

  const int k = k_pages < visible ? k_pages : visible;
  const std::vector<int> ids = select_pages(scores, k, n_recent);
  if (selected != nullptr) {
    for (int i = 0; i < k_pages; ++i) selected[i] = i < static_cast<int>(ids.size()) ? ids[static_cast<size_t>(i)] : -1;
  }

  std::vector<const float*> key_rows;
  std::vector<const float*> value_rows;
  for (int page : ids) {
    if (page < 0 || page >= visible) continue;
    int len = kv.page_lens[page];
    if (len > page_size) len = page_size;
    for (int local = 0; local < len; ++local) {
      const int pos = page * page_size + local;
      if (pos > query_position) continue;
      const size_t offset = (static_cast<size_t>(page) * page_size + local) * dim;
      key_rows.push_back(kv.keys + offset);
      value_rows.push_back(kv.values + offset);
    }
  }
  softmax_weighted_sum(query, key_rows, value_rows, dim, scale, out);
}

}  // namespace tinyserve
