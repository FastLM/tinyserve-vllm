#include "tinyserve/page_cache.hpp"

#include <stdexcept>

namespace tinyserve {

void reduce_bounds(const float* keys, int count, int dim, float* k_min, float* k_max) {
  if (keys == nullptr || k_min == nullptr || k_max == nullptr || dim <= 0 || count < 0) {
    throw std::invalid_argument("reduce_bounds received an invalid buffer");
  }
  if (count == 0) {
    for (int d = 0; d < dim; ++d) {
      k_min[d] = 0.f;
      k_max[d] = 0.f;
    }
    return;
  }
  for (int d = 0; d < dim; ++d) {
    k_min[d] = keys[d];
    k_max[d] = keys[d];
  }
  for (int token = 1; token < count; ++token) {
    const float* row = keys + static_cast<size_t>(token) * dim;
    for (int d = 0; d < dim; ++d) {
      if (row[d] < k_min[d]) k_min[d] = row[d];
      if (row[d] > k_max[d]) k_max[d] = row[d];
    }
  }
}

void update_bounds(const float* key, int dim, float* k_min, float* k_max, bool reset) {
  if (key == nullptr || k_min == nullptr || k_max == nullptr || dim <= 0) {
    throw std::invalid_argument("update_bounds received an invalid buffer");
  }
  if (reset) {
    for (int d = 0; d < dim; ++d) {
      k_min[d] = key[d];
      k_max[d] = key[d];
    }
    return;
  }
  for (int d = 0; d < dim; ++d) {
    if (key[d] < k_min[d]) k_min[d] = key[d];
    if (key[d] > k_max[d]) k_max[d] = key[d];
  }
}

}  // namespace tinyserve
