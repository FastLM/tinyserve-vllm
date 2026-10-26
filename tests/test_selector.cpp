#include "tinyserve/bounding_box.hpp"
#include "tinyserve/memory_model.hpp"
#include "tinyserve/page_cache.hpp"
#include "tinyserve/topk.hpp"

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

std::vector<int> ids_of(const std::vector<tinyserve::ScoredPage>& pages) {
  std::vector<int> ids;
  for (const auto& page : pages) ids.push_back(page.id);
  return ids;
}

}  // namespace

int main() {
  const float query[4] = {1.f, -2.f, 0.f, 0.5f};
  const float k_min[4] = {-1.f, -3.f, 0.f, 1.f};
  const float k_max[4] = {2.f, 4.f, 1.f, 3.f};
  const float score = tinyserve::bounding_box_score(query, k_min, k_max, 4);
  expect(std::fabs(score - 9.5f) < 1e-5f, "hand bounding-box score");

  const float keys[] = {2.f, -3.f, 0.5f, 3.f, 0.f, 0.f, 0.f, 0.f, -1.f, 4.f, 0.f, 1.f};
  float lower[4];
  float upper[4];
  tinyserve::reduce_bounds(keys, 3, 4, lower, upper);
  const float bound = tinyserve::bounding_box_score(query, lower, upper, 4);
  float max_dot = -1e30f;
  for (int row = 0; row < 3; ++row) {
    float dot = 0.f;
    for (int d = 0; d < 4; ++d) dot += query[d] * keys[row * 4 + d];
    if (dot > max_dot) max_dot = dot;
  }
  expect(bound + 1e-4f >= max_dot, "bounding box upper-bounds every key");

  float running_min[4];
  float running_max[4];
  tinyserve::update_bounds(keys, 4, running_min, running_max, true);
  tinyserve::update_bounds(keys + 4, 4, running_min, running_max, false);
  tinyserve::update_bounds(keys + 8, 4, running_min, running_max, false);
  for (int d = 0; d < 4; ++d) {
    expect(running_min[d] == lower[d] && running_max[d] == upper[d], "incremental bounds match reduce");
  }

  const std::vector<float> tied = {1.f, 2.f, 2.f, 0.f, 2.f};
  const auto top = tinyserve::streaming_topk(tied, {0, 1, 2, 3, 4}, 4, 8);
  const std::vector<int> top_ids = ids_of(top);
  expect(top_ids == std::vector<int>({1, 2, 4, 0}), "tie break prefers the smaller page");

  const auto short_top = tinyserve::streaming_topk({0.2f, 1.f, -3.f}, {0, 1, 2}, 5, 4);
  expect(ids_of(short_top) == std::vector<int>({1, 0, 2, -1, -1}), "short list is padded with -1");

  const std::vector<float> pages = {10.f, 0.2f, 0.3f, 0.1f};
  const auto pinned = tinyserve::select_pages(pages, 1, 1, 4);
  expect(pinned.size() == 1 && pinned[0] == 3, "k=1 keeps the newest page");

  const double length = 32768.0;
  const double page = 16.0;
  const double selected = 128.0;
  const double reuse = 0.5;
  const double fraction = tinyserve::memory_fraction(length, page, selected, reuse);
  const double load = tinyserve::memory_load(length, page, selected, reuse, 4.0);
  expect(std::fabs(load / (2.0 * 4.0 * length) - fraction) < 1e-12, "load formula matches the fraction");
  const double stated = tinyserve::paper_page_size(length, selected);
  expect(std::fabs(tinyserve::memory_fraction(length, stated, selected, reuse) -
                   tinyserve::fraction_at_paper_page_size(length, selected, reuse)) < 1e-9,
         "paper page size");
  const double best = tinyserve::minimizing_page_size(4096.0, 32.0, 0.4);
  const double best_fraction = tinyserve::minimized_memory_fraction(4096.0, 32.0, 0.4);
  expect(std::fabs(tinyserve::memory_fraction(4096.0, best, 32.0, 0.4) - best_fraction) < 1e-9,
         "minimizing page size");

  if (failures != 0) {
    std::cerr << failures << " selector checks failed\n";
    return 1;
  }
  std::cout << "test_selector ok\n";
  return 0;
}
