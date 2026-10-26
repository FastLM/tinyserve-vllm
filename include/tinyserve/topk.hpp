#pragma once

#include <vector>

namespace tinyserve {

struct ScoredPage {
  float score;
  int id;
};

bool rank_better(ScoredPage left, ScoredPage right);

// Descending bitonic sort. The length must be a power of two.
void bitonic_desc(std::vector<ScoredPage>& values);

// Tile-wise top-k used by the fused CUDA kernel. Best page first.
// Missing slots are (-inf, -1) when k is larger than the input.
std::vector<ScoredPage> streaming_topk(const std::vector<float>& scores,
                                       const std::vector<int>& ids, int k, int tile = 128);

// Force the last n_recent page ids into a best-first selection.
std::vector<int> pin_recent(std::vector<int> selected, int num_pages, int n_recent);

// streaming_topk followed by pin_recent. Page ids are 0..n-1.
std::vector<int> select_pages(const std::vector<float>& scores, int k, int n_recent, int tile = 128);

}  // namespace tinyserve
