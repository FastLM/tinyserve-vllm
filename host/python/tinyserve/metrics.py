"""Cost model from the query-aware page-selection analysis.

Symbols match the paper:

- ``L`` cache length in tokens
- ``S`` page size
- ``K`` number of selected pages
- ``rho`` probability a selected page is already resident and is not reloaded
- ``M`` bytes per token

Per-step traffic counts both key and value::

    Load = 2 M (L / S + rho K S)

Relative to a full KV reload ``2 M L``::

    fraction = 1 / S + rho K S / L

The page size stated in the paper is ``S* = sqrt(L / K)``. Substituting that
choice gives ``(1 + rho) sqrt(K / L)``. Differentiating the fraction with
respect to ``S`` puts the minimum at ``sqrt(L / (rho K))``, where the fraction
is ``2 sqrt(rho K / L)``.
"""

from __future__ import annotations

import math


def memory_load(
    length: float,
    page_size: float,
    k_pages: float,
    reuse: float,
    bytes_per_token: float,
) -> float:
    """Bytes moved per decode step, including key and value."""
    if page_size <= 0 or length < 0:
        raise ValueError("length must be non-negative and page_size positive")
    return 2.0 * bytes_per_token * (length / page_size + reuse * k_pages * page_size)


def memory_fraction(length: float, page_size: float, k_pages: float, reuse: float) -> float:
    """KV traffic relative to scanning the full cache."""
    if length <= 0 or page_size <= 0:
        raise ValueError("length and page_size must be positive")
    return (1.0 / page_size) + reuse * (k_pages * page_size) / length


def paper_page_size(length: float, k_pages: float) -> float:
    """``S* = sqrt(L / K)`` as stated for the memory-fraction bound."""
    if length <= 0 or k_pages <= 0:
        raise ValueError("length and k_pages must be positive")
    return math.sqrt(length / k_pages)


def fraction_at_paper_page_size(length: float, k_pages: float, reuse: float) -> float:
    """Fraction at ``S* = sqrt(L / K)``: ``(1 + rho) sqrt(K / L)``."""
    if length <= 0 or k_pages <= 0:
        raise ValueError("length and k_pages must be positive")
    return (1.0 + reuse) * math.sqrt(k_pages / length)


def minimizing_page_size(length: float, k_pages: float, reuse: float) -> float:
    """Page size that minimizes ``1/S + rho K S / L``."""
    if length <= 0 or k_pages <= 0 or reuse <= 0:
        raise ValueError("length, k_pages, and reuse must be positive")
    return math.sqrt(length / (reuse * k_pages))


def minimized_memory_fraction(length: float, k_pages: float, reuse: float) -> float:
    """Minimum of the memory fraction: ``2 sqrt(rho K / L)``."""
    if length <= 0 or k_pages <= 0 or reuse <= 0:
        raise ValueError("length, k_pages, and reuse must be positive")
    return 2.0 * math.sqrt(reuse * k_pages / length)


def approximation_gap_bound(head_dim: float, variance: float, page_size: int) -> float:
    """Paper bound on the expected gap between the true max dot and the box score.

    The directional box score is an upper bound, so the signed error
    ``max_k q·k - r(q, phi)`` is <= 0. The published inequality bounds the
    magnitude of that gap::

        E[gap] <= d * sigma^2 / S * sqrt(log S)
    """
    if page_size <= 1:
        raise ValueError("page_size must be greater than 1")
    if head_dim < 0 or variance < 0:
        raise ValueError("head_dim and variance must be non-negative")
    return head_dim * variance / page_size * math.sqrt(math.log(page_size))
