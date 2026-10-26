#pragma once

namespace tinyserve {

// Directional bounding box. This is an upper bound on max_k q·k inside the page.
float bounding_box_score(const float* query, const float* k_min, const float* k_max, int dim);

}  // namespace tinyserve
