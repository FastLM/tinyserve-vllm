"""Optional CUDA extension for the fused decode kernel.

The reference implementation in :mod:`tinyserve.attention` is what training
and CPU inference run. When ``tinyserve_cuda`` is importable and the decode
step matches the kernel's shape contract, the serving path uses the fused
kernel instead.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

MAX_K = 256
MAX_RECENT = 32
HEAD_DIMS = (32, 64, 128)
PAGE_SIZES = (8, 16, 32, 64)

_EXTENSION = None
_TRIED = False


def load_extension():
    """Import the compiled extension once. Missing extension returns ``None``."""
    global _EXTENSION, _TRIED
    if _TRIED:
        return _EXTENSION
    _TRIED = True
    try:
        import tinyserve_cuda  # type: ignore
    except ImportError:
        _EXTENSION = None
    else:
        _EXTENSION = tinyserve_cuda
    return _EXTENSION


def supports(head_dim: int, page_size: int, k_pages: int, n_recent: int) -> bool:
    return (
        head_dim in HEAD_DIMS
        and page_size in PAGE_SIZES
        and 0 < k_pages <= MAX_K
        and 0 <= n_recent <= MAX_RECENT
    )


def fused_decode(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    k_min: torch.Tensor,
    k_max: torch.Tensor,
    page_lens: torch.Tensor,
    num_pages: torch.Tensor,
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
    k_pages_k: int,
    n_recent: int,
    scale: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Run the fused kernel for a single decode token.

    ``q`` is ``[B, Hq, D]``. KV pages are ``[B, Hkv, P, S, D]`` in fp16 or
    int8. Returns ``(out [B, Hq, D], selected [B, Hq, K])`` or ``None`` when
    the extension or the shape is unavailable. Callers fall back to the
    PyTorch reference in that case.
    """
    ext = load_extension()
    if ext is None:
        return None
    if q.dim() != 3 or k_pages.dim() != 5:
        return None
    head_dim = q.shape[-1]
    page_size = k_pages.shape[-2]
    if not supports(head_dim, page_size, k_pages_k, n_recent):
        return None
    if not q.is_cuda or not k_pages.is_cuda:
        return None
    if k_pages.dtype not in (torch.float16, torch.int8):
        return None
    if q.requires_grad or k_pages.requires_grad or v_pages.requires_grad:
        return None

    empty = torch.empty(0, device=q.device, dtype=torch.float32)
    out, selected = ext.fused_sparse_attention(
        q.float().contiguous(),
        k_pages.contiguous(),
        v_pages.contiguous(),
        k_min.float().contiguous(),
        k_max.float().contiguous(),
        page_lens.to(dtype=torch.int32).contiguous(),
        num_pages.to(dtype=torch.int32).contiguous(),
        empty if k_scale is None else k_scale.float().contiguous(),
        empty if v_scale is None else v_scale.float().contiguous(),
        int(k_pages_k),
        int(n_recent),
        float(scale),
    )
    return out, selected.to(dtype=torch.long)
