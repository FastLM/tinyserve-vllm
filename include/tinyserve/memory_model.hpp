#pragma once

#include <cmath>
#include <stdexcept>

namespace tinyserve {

// Load = 2 M (L/S + rho K S). fraction = 1/S + rho K S / L.
inline double memory_load(double length, double page_size, double k_pages, double reuse,
                          double bytes_per_token) {
  if (page_size <= 0.0 || length < 0.0) {
    throw std::invalid_argument("length must be non-negative and page_size positive");
  }
  return 2.0 * bytes_per_token * (length / page_size + reuse * k_pages * page_size);
}

inline double memory_fraction(double length, double page_size, double k_pages, double reuse) {
  if (length <= 0.0 || page_size <= 0.0) {
    throw std::invalid_argument("length and page_size must be positive");
  }
  return (1.0 / page_size) + reuse * (k_pages * page_size) / length;
}

inline double paper_page_size(double length, double k_pages) {
  if (length <= 0.0 || k_pages <= 0.0) {
    throw std::invalid_argument("length and k_pages must be positive");
  }
  return std::sqrt(length / k_pages);
}

inline double fraction_at_paper_page_size(double length, double k_pages, double reuse) {
  if (length <= 0.0 || k_pages <= 0.0) {
    throw std::invalid_argument("length and k_pages must be positive");
  }
  return (1.0 + reuse) * std::sqrt(k_pages / length);
}

inline double minimizing_page_size(double length, double k_pages, double reuse) {
  if (length <= 0.0 || k_pages <= 0.0 || reuse <= 0.0) {
    throw std::invalid_argument("length, k_pages, and reuse must be positive");
  }
  return std::sqrt(length / (reuse * k_pages));
}

inline double minimized_memory_fraction(double length, double k_pages, double reuse) {
  if (length <= 0.0 || k_pages <= 0.0 || reuse <= 0.0) {
    throw std::invalid_argument("length, k_pages, and reuse must be positive");
  }
  return 2.0 * std::sqrt(reuse * k_pages / length);
}

}  // namespace tinyserve
