#include "tinyserve.h"

#include <cstdio>

// One decode step over two pages. Page 1 is the newest page and is pinned
// even though page 0 has the larger bounding-box score.
int main() {
  constexpr int kDim = 4;
  constexpr int kPage = 2;
  const float query[kDim] = {1.f, -2.f, 0.f, 0.5f};
  const float keys[2 * kPage * kDim] = {
      2.f, -3.f, 0.f, 3.f, 0.f, 0.f, 0.f, 0.f, 0.2f, -0.1f, 0.f, 0.4f, 0.f, 0.f, 0.f, 0.f,
  };
  const float values[2 * kPage * kDim] = {
      1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f,
  };
  float k_min[2 * kDim];
  float k_max[2 * kDim];
  const int page_lens[2] = {2, 1};
  for (int page = 0; page < 2; ++page) {
    tinyserve_reduce_bounds(keys + page * kPage * kDim, page_lens[page], kDim, k_min + page * kDim,
                            k_max + page * kDim);
  }

  float score0 = 0.f;
  float score1 = 0.f;
  tinyserve_bounding_box_score(query, k_min, k_max, kDim, &score0);
  tinyserve_bounding_box_score(query, k_min + kDim, k_max + kDim, kDim, &score1);

  TinyServeKvView kv;
  kv.keys = keys;
  kv.values = values;
  kv.k_min = k_min;
  kv.k_max = k_max;
  kv.page_lens = page_lens;
  kv.num_pages = 2;
  kv.page_size = kPage;
  kv.dim = kDim;

  float out[kDim] = {};
  int selected[1] = {-1};
  const int rc = tinyserve_sparse_decode(query, &kv, 1, 1, 1.f, 2, out, selected);
  std::printf("page scores: %.4f %.4f\n", score0, score1);
  std::printf("selected page: %d\n", selected[0]);
  std::printf("output: %.4f %.4f %.4f %.4f\n", out[0], out[1], out[2], out[3]);
  return rc == 0 && selected[0] == 1 ? 0 : 1;
}
