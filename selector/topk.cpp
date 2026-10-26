#include "tinyserve/topk.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace tinyserve {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

}  // namespace

bool rank_better(ScoredPage left, ScoredPage right) {
  if (left.score != right.score) return left.score > right.score;
  return left.id < right.id;
}

void bitonic_desc(std::vector<ScoredPage>& values) {
  const int n = static_cast<int>(values.size());
  if (n <= 0 || (n & (n - 1)) != 0) {
    throw std::invalid_argument("bitonic sort expects a non-empty power-of-two length");
  }
  for (int span = 2; span <= n; span <<= 1) {
    for (int stride = span >> 1; stride > 0; stride >>= 1) {
      for (int i = 0; i < n; ++i) {
        const int partner = i ^ stride;
        if (partner <= i) continue;
        const bool left_better = rank_better(values[i], values[partner]);
        if ((i & span) == 0) {
          if (!left_better) std::swap(values[i], values[partner]);
        } else if (left_better) {
          std::swap(values[i], values[partner]);
        }
      }
    }
  }
}

std::vector<ScoredPage> merge_topk(const std::vector<ScoredPage>& top,
                                   const std::vector<ScoredPage>& tile, int k) {
  std::vector<ScoredPage> merged;
  merged.reserve(static_cast<size_t>(k));
  int i = 0;
  int j = 0;
  while (static_cast<int>(merged.size()) < k) {
    if (i >= static_cast<int>(top.size())) {
      merged.push_back(tile[static_cast<size_t>(j++)]);
    } else if (j >= static_cast<int>(tile.size())) {
      merged.push_back(top[static_cast<size_t>(i++)]);
    } else if (rank_better(top[static_cast<size_t>(i)], tile[static_cast<size_t>(j)])) {
      merged.push_back(top[static_cast<size_t>(i++)]);
    } else if (rank_better(tile[static_cast<size_t>(j)], top[static_cast<size_t>(i)])) {
      merged.push_back(tile[static_cast<size_t>(j++)]);
    } else {
      merged.push_back(top[static_cast<size_t>(i++)]);
    }
  }
  return merged;
}

std::vector<ScoredPage> streaming_topk(const std::vector<float>& scores, const std::vector<int>& ids,
                                       int k, int tile) {
  if (tile <= 0 || (tile & (tile - 1)) != 0) {
    throw std::invalid_argument("tile must be a positive power of two");
  }
  if (scores.size() != ids.size()) {
    throw std::invalid_argument("scores and ids must have the same length");
  }
  if (k <= 0) return {};

  const int n = static_cast<int>(scores.size());
  const int pad = (tile - (n % tile)) % tile;
  std::vector<ScoredPage> pairs;
  pairs.reserve(static_cast<size_t>(n + pad));
  for (int i = 0; i < n; ++i) pairs.push_back(ScoredPage{scores[static_cast<size_t>(i)], ids[static_cast<size_t>(i)]});
  for (int i = 0; i < pad; ++i) pairs.push_back(ScoredPage{kNegInf, -1});

  std::vector<ScoredPage> top(static_cast<size_t>(k), ScoredPage{kNegInf, -1});
  for (int base = 0; base < static_cast<int>(pairs.size()); base += tile) {
    std::vector<ScoredPage> chunk(pairs.begin() + base, pairs.begin() + base + tile);
    bitonic_desc(chunk);
    top = merge_topk(top, chunk, k);
  }
  return top;
}

std::vector<int> pin_recent(std::vector<int> selected, int num_pages, int n_recent) {
  const int k = static_cast<int>(selected.size());
  if (num_pages <= 0 || n_recent <= 0 || k == 0) return selected;

  const int recent_lo = num_pages > n_recent ? num_pages - n_recent : 0;
  std::vector<int> missing;
  for (int page = recent_lo; page < num_pages; ++page) {
    bool found = false;
    for (int id : selected) {
      if (id == page) {
        found = true;
        break;
      }
    }
    if (!found) missing.push_back(page);
  }
  if (missing.empty()) return selected;

  std::vector<int> evictable;
  for (int i = k - 1; i >= 0; --i) {
    const int id = selected[static_cast<size_t>(i)];
    if (id < recent_lo || id >= num_pages) evictable.push_back(i);
  }
  if (static_cast<int>(missing.size()) > static_cast<int>(evictable.size())) {
    const int start = num_pages > k ? num_pages - k : 0;
    std::vector<int> out;
    out.reserve(static_cast<size_t>(k));
    for (int page = start; page < num_pages && static_cast<int>(out.size()) < k; ++page) out.push_back(page);
    while (static_cast<int>(out.size()) < k) out.push_back(-1);
    return out;
  }
  for (size_t i = 0; i < missing.size(); ++i) {
    selected[static_cast<size_t>(evictable[i])] = missing[i];
  }
  return selected;
}

std::vector<int> select_pages(const std::vector<float>& scores, int k, int n_recent, int tile) {
  std::vector<int> ids(scores.size());
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int>(i);
  const auto top = streaming_topk(scores, ids, k, tile);
  std::vector<int> chosen;
  chosen.reserve(top.size());
  for (const ScoredPage& page : top) chosen.push_back(page.id);
  return pin_recent(std::move(chosen), static_cast<int>(scores.size()), n_recent);
}

}  // namespace tinyserve
