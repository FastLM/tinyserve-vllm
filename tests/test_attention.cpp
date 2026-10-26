#include "tinyserve/page_cache.hpp"
#include "tinyserve/sparse_attention.hpp"
#include "tinyserve.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expect(bool cond, const char* message) {
  if (!cond) {
    std::cerr << "FAIL " << message << "\n";
    ++failures;
  }
}

bool close(const std::vector<float>& left, const std::vector<float>& right) {
  if (left.size() != right.size()) return false;
  for (size_t i = 0; i < left.size(); ++i) {
    if (std::fabs(left[i] - right[i]) > 1e-5f) return false;
  }
  return true;
}

}  // namespace

int main() {
  constexpr int kDim = 2;
  constexpr int kPage = 2;
  constexpr int kTokens = 3;
  const float query[kDim] = {1.f, 0.f};
  const float token_keys[kTokens * kDim] = {1.f, 0.f, 0.f, 1.f, 1.f, 0.f};
  const float token_values[kTokens * kDim] = {4.f, 0.f, 0.f, 5.f, 6.f, 1.f};
  std::vector<float> dense(kDim);
  tinyserve::dense_decode(query, token_keys, token_values, kTokens, kDim, 1.f, dense.data());

  float page_keys[2 * kPage * kDim] = {};
  float page_values[2 * kPage * kDim] = {};
  for (int token = 0; token < kTokens; ++token) {
    for (int d = 0; d < kDim; ++d) {
      page_keys[(token * kDim) + d] = token_keys[token * kDim + d];
      page_values[(token * kDim) + d] = token_values[token * kDim + d];
    }
  }
  page_keys[3 * kDim + 0] = 99.f;
  page_values[3 * kDim + 1] = 99.f;
  const int page_lens[2] = {2, 1};
  float k_min[2 * kDim];
  float k_max[2 * kDim];
  for (int page = 0; page < 2; ++page) {
    tinyserve::reduce_bounds(page_keys + page * kPage * kDim, page_lens[page], kDim, k_min + page * kDim,
                             k_max + page * kDim);
  }

  tinyserve::KvPageView kv;
  kv.keys = page_keys;
  kv.values = page_values;
  kv.k_min = k_min;
  kv.k_max = k_max;
  kv.page_lens = page_lens;
  kv.num_pages = 2;
  kv.page_size = kPage;
  kv.dim = kDim;
  std::vector<float> sparse(kDim);
  int selected[2] = {-1, -1};
  tinyserve::sparse_decode(query, kv, 2, 1, 1.f, kTokens - 1, sparse.data(), selected);
  expect(close(dense, sparse), "full budget matches dense decode");
  expect(selected[0] != -1 && selected[1] != -1, "both pages are selected");

  int only_recent[1] = {-1};
  tinyserve::sparse_decode(query, kv, 1, 1, 1.f, kTokens - 1, sparse.data(), only_recent);
  expect(only_recent[0] == 1, "tight budget keeps the current page");

  TinyServeKvView c_view;
  c_view.keys = page_keys;
  c_view.values = page_values;
  c_view.k_min = k_min;
  c_view.k_max = k_max;
  c_view.page_lens = page_lens;
  c_view.num_pages = 2;
  c_view.page_size = kPage;
  c_view.dim = kDim;
  float c_out[kDim] = {};
  int c_selected[2] = {};
  const int rc = tinyserve_sparse_decode(query, &c_view, 2, 0, 1.f, kTokens - 1, c_out, c_selected);
  expect(rc == 0, "C API returns success");
  expect(close(dense, {c_out[0], c_out[1]}), "C API matches dense decode");

  if (failures != 0) {
    std::cerr << failures << " attention checks failed\n";
    return 1;
  }
  std::cout << "test_attention ok\n";
  return 0;
}
