#include "tinyserve/bounding_box.hpp"

#include <stdexcept>

namespace tinyserve {

float bounding_box_score(const float* query, const float* k_min, const float* k_max, int dim) {
  if (query == nullptr || k_min == nullptr || k_max == nullptr || dim <= 0) {
    throw std::invalid_argument("bounding_box_score received an empty vector");
  }
  float score = 0.f;
  for (int i = 0; i < dim; ++i) {
    const float q = query[i];
    score += q * (q >= 0.f ? k_max[i] : k_min[i]);
  }
  return score;
}

}  // namespace tinyserve
