#pragma once

namespace tinyserve {

// Channel-wise min/max over keys laid out as [count, dim].
void reduce_bounds(const float* keys, int count, int dim, float* k_min, float* k_max);

// Fold one new key into an existing page. reset replaces the bounds.
void update_bounds(const float* key, int dim, float* k_min, float* k_max, bool reset);

}  // namespace tinyserve
