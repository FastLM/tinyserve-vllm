"""Gradient-aware page retention and optional distributed grad sync.

Sparse attention saves the selected pages for the backward, so pages that no
query fetched receive zero value gradient. ``page_importance`` turns that
gradient into a per-page score. Fine-tuning can checkpoint the decoder stack
(``TinyServeConfig.checkpoint``) and keep only the high-scoring pages.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F


def page_importance(grad: torch.Tensor, page_size: int) -> torch.Tensor:
    """L2 norm of ``dL/dV`` inside each page. ``grad`` is ``[B, H, T, D]``."""
    if grad.dim() != 4:
        raise ValueError("grad must have shape [batch, heads, tokens, dim]")
    batch, heads, tokens, dim = grad.shape
    slots = max(1, (tokens + page_size - 1) // page_size)
    pad = slots * page_size - tokens
    if pad:
        grad = F.pad(grad, (0, 0, 0, pad))
    pages = grad.float().reshape(batch, heads, slots, page_size, dim)
    return torch.linalg.vector_norm(pages, dim=(-1, -2))


def retain_mask(importance: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Boolean mask of the highest-importance pages. ``importance`` is ``[B, H, P]``."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    num_pages = importance.shape[-1]
    keep = max(1, int(round(keep_ratio * num_pages)))
    keep = min(keep, num_pages)
    index = torch.topk(importance, keep, dim=-1).indices
    mask = torch.zeros_like(importance, dtype=torch.bool)
    return mask.scatter(-1, index, torch.ones_like(index, dtype=torch.bool))


class AsyncGradSync:
    """All-reduce parameter gradients when ``torch.distributed`` is initialized.

    A single process, or a process group that was never initialized, is a no-op
    so the same training step runs on one GPU and on a multi-GPU job.
    """

    def sync(self, params: Iterable[torch.nn.Parameter]) -> None:
        dist = _distributed()
        if dist is None or not dist.is_initialized():
            return
        world = dist.get_world_size()
        work = []
        live = [param for param in params if param.grad is not None]
        for param in live:
            work.append(dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True))
        for handle in work:
            handle.wait()
        if world > 1:
            for param in live:
                param.grad.div_(world)


def _distributed():
    try:
        import torch.distributed as dist
    except ImportError:
        return None
    if not dist.is_available():
        return None
    return dist
