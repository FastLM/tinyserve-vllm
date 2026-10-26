"""Dense causal attention and query-aware sparse attention.

The relevance of page ``j`` with per-channel key bounds ``(m_j, M_j)`` is

    r(q, phi_j) = sum_i q_i * (M_{j,i} if q_i >= 0 else m_{j,i})

which is an upper bound on ``max_{k in page j} q·k``. Decode gathers the
top-scoring pages, always keeps the newest pages, and runs softmax over the
tokens inside that union. Training uses the same math on the whole sequence.
The fused CUDA kernel is an inference fast path for a single new token.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .cuda_backend import fused_decode


def expand_kv(states: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Repeat KV heads so they line up with query heads."""
    num_kv = states.shape[1]
    if num_kv == num_q_heads:
        return states
    if num_q_heads % num_kv != 0:
        raise ValueError(f"cannot map {num_kv} KV heads onto {num_q_heads} query heads")
    return states.repeat_interleave(num_q_heads // num_kv, dim=1)


def bounding_box_scores(query: torch.Tensor, k_min: torch.Tensor, k_max: torch.Tensor) -> torch.Tensor:
    """Directional bounding-box score.

    ``query`` is ``[B, H, T, D]`` and the bounds are ``[B, H, P, D]``.
    The result is ``[B, H, T, P]``.
    """
    positive = query >= 0
    upper = k_max.unsqueeze(-3)
    lower = k_min.unsqueeze(-3)
    bound = torch.where(positive.unsqueeze(-2), upper, lower)
    return (query.unsqueeze(-2) * bound).sum(dim=-1)


def page_lengths(
    lengths: torch.Tensor,
    num_page_slots: int,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Valid tokens in each page, and how many pages each sequence uses."""
    lengths = lengths.to(dtype=torch.long)
    device = lengths.device
    slots = torch.arange(num_page_slots, device=device)
    full = torch.div(lengths, page_size, rounding_mode="floor")
    remainder = lengths % page_size
    lens = torch.zeros(lengths.shape[0], num_page_slots, dtype=torch.long, device=device)
    lens = torch.where(slots.view(1, -1) < full.view(-1, 1), torch.full_like(lens, page_size), lens)
    partial = (remainder > 0).view(-1, 1) & (slots.view(1, -1) == full.view(-1, 1))
    lens = torch.where(partial, remainder.view(-1, 1).expand_as(lens), lens)
    num_pages = torch.where(
        lengths == 0,
        torch.zeros_like(full),
        torch.where(remainder == 0, full, full + 1),
    )
    return lens, num_pages


def page_bounds(pages: torch.Tensor, page_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Channel-wise min and max of the valid keys in each page.

    ``pages`` is ``[B, H, P, S, D]``. Empty pages get zeros so a later mask,
    rather than an infinite bound, decides whether they can be selected.
    """
    batch, heads, _, page_size, _ = pages.shape
    local = torch.arange(page_size, device=pages.device)
    valid = local.view(1, 1, 1, page_size) < page_lens.view(batch, 1, -1, 1)
    valid = valid.unsqueeze(-1)
    lower = pages.masked_fill(~valid, float("inf")).amin(dim=-2)
    upper = pages.masked_fill(~valid, float("-inf")).amax(dim=-2)
    empty = (page_lens == 0).view(batch, 1, -1, 1)
    lower = lower.masked_fill(empty, 0)
    upper = upper.masked_fill(empty, 0)
    return lower, upper


def paginate(states: torch.Tensor, page_size: int, lengths: torch.Tensor):
    """Pad ``[B, H, T, D]`` into ``[B, H, P, S, D]`` plus lengths and bounds."""
    batch, heads, tokens, dim = states.shape
    num_slots = max(1, math.ceil(tokens / page_size)) if tokens else 1
    pad = num_slots * page_size - tokens
    if pad:
        states = F.pad(states, (0, 0, 0, pad))
    elif tokens == 0:
        states = states.new_zeros(batch, heads, num_slots * page_size, dim)
    pages = states.view(batch, heads, num_slots, page_size, dim)
    lens, num_pages = page_lengths(lengths, num_slots, page_size)
    return pages, lens, num_pages


def select_pages(
    scores: torch.Tensor,
    page_lens: torch.Tensor,
    num_pages: torch.Tensor,
    query_positions: torch.Tensor,
    page_size: int,
    k_pages: int,
    n_recent: int,
) -> torch.Tensor:
    """Pick page ids ``[B, H, T, K]``. Invalid slots are ``-1``.

    Visible recent pages outrank the rest. A stable score sort, followed by a
    stable priority sort, keeps the newest page when several pages are pinned.
    A float bonus near 1e30 cannot encode that age because consecutive integers
    collapse.
    """
    batch, heads, queries, num_slots = scores.shape
    device = scores.device
    page_index = torch.arange(num_slots, device=device)
    page_start = page_index * page_size
    positions = query_positions.to(device=device, dtype=torch.long)
    causal = page_start.view(1, 1, 1, num_slots) <= positions.view(batch, 1, queries, 1)
    nonempty = page_lens.view(batch, 1, 1, num_slots) > 0
    in_sequence = page_index.view(1, 1, 1, num_slots) < num_pages.view(batch, 1, 1, 1)
    visible = causal & nonempty & in_sequence
    masked = scores.masked_fill(~visible, float("-inf"))
    priority = torch.zeros_like(masked)
    if n_recent > 0:
        last = torch.minimum(
            positions // page_size,
            (num_pages - 1).clamp_min(0).unsqueeze(1),
        )
        alive = num_pages > 0
        head_index = torch.arange(heads, device=device)
        for age in range(n_recent):
            page = last - age
            ok = (page >= 0) & alive.unsqueeze(1)
            batch_index, query_index = torch.where(ok)
            if batch_index.numel() == 0:
                continue
            page_ids = page[batch_index, query_index]
            keep = visible[batch_index, 0, query_index, page_ids]
            batch_index = batch_index[keep]
            query_index = query_index[keep]
            page_ids = page_ids[keep]
            if batch_index.numel() == 0:
                continue
            level = masked.new_tensor(float(n_recent - age))
            rows = batch_index.shape[0]
            priority[
                batch_index[:, None].expand(rows, heads).reshape(-1),
                head_index[None, :].expand(rows, heads).reshape(-1),
                query_index[:, None].expand(rows, heads).reshape(-1),
                page_ids[:, None].expand(rows, heads).reshape(-1),
            ] = level
    k = min(k_pages, num_slots)
    if k <= 0:
        return torch.full((batch, heads, queries, 0), -1, dtype=torch.long, device=device)
    by_score = torch.argsort(masked, dim=-1, descending=True, stable=True)
    priority_in_score_order = torch.gather(priority, -1, by_score)
    by_priority = torch.argsort(priority_in_score_order, dim=-1, descending=True, stable=True)
    chosen = torch.gather(by_score, -1, by_priority)[..., :k]
    chosen_score = torch.gather(masked, -1, chosen)
    return torch.where(torch.isfinite(chosen_score), chosen, torch.full_like(chosen, -1))


def _dequantize(pages: torch.Tensor, scale: Optional[torch.Tensor]) -> torch.Tensor:
    if scale is None:
        return pages
    return pages.float() * scale[:, :, :, None, None]


def _cuda_decode_ok(
    query: torch.Tensor,
    k_pages: torch.Tensor,
    page_lens: torch.Tensor,
    num_pages: torch.Tensor,
    query_positions: torch.Tensor,
    page_size: int,
    prune_threshold: float,
    approx: bool,
) -> bool:
    """The fused kernel scores every resident page, so the query must be the tail."""
    if approx or prune_threshold > 0 or query.shape[2] != 1:
        return False
    if query.requires_grad or k_pages.requires_grad:
        return False
    if not query.is_cuda or num_pages.numel() == 0:
        return False
    if not bool((num_pages > 0).all()):
        return False
    last = (num_pages - 1).clamp_min(0)
    batch = torch.arange(num_pages.shape[0], device=page_lens.device)
    last_len = page_lens[batch, last]
    tail = last * page_size + last_len - 1
    positions = query_positions[:, 0].to(device=tail.device, dtype=torch.long)
    return bool(torch.equal(positions, tail))


def sparse_cached_attention(
    query: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    k_min: torch.Tensor,
    k_max: torch.Tensor,
    page_lens: torch.Tensor,
    num_pages: torch.Tensor,
    *,
    page_size: int,
    k_pages_k: int,
    n_recent: int,
    scale: float,
    query_positions: torch.Tensor,
    prune_threshold: float = 0.0,
    approx: bool = False,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention over pages that already live in a KV cache.

    ``query`` is ``[B, Hq, Tq, D]``. Pages are ``[B, Hkv, P, S, D]``.
    ``query_positions`` gives the absolute token index of each query and is
    ``[B, Tq]``.
    """
    out_dtype = query.dtype
    batch, num_q_heads, queries, _ = query.shape
    device = query.device
    page_lens = page_lens.to(device=device, dtype=torch.long)
    num_pages = num_pages.to(device=device, dtype=torch.long)
    query_positions = query_positions.to(device=device, dtype=torch.long)
    if query_positions.shape != (batch, queries):
        raise ValueError("query_positions must have shape [batch, queries]")

    if _cuda_decode_ok(
        query, k_pages, page_lens, num_pages, query_positions, page_size, prune_threshold, approx
    ):
        fused = fused_decode(
            query[:, :, 0, :],
            k_pages,
            v_pages,
            k_min,
            k_max,
            page_lens,
            num_pages,
            k_scale,
            v_scale,
            k_pages_k,
            n_recent,
            scale,
        )
        if fused is not None:
            out, selected = fused
            return out.unsqueeze(2).to(dtype=out_dtype), selected.unsqueeze(2)

    k_vals = _dequantize(k_pages, k_scale)
    v_vals = _dequantize(v_pages, v_scale)
    k_vals = expand_kv(k_vals, num_q_heads).float()
    v_vals = expand_kv(v_vals, num_q_heads).float()
    bound_min = expand_kv(k_min.float(), num_q_heads)
    bound_max = expand_kv(k_max.float(), num_q_heads)
    q = query.float()
    with torch.no_grad():
        scores = bounding_box_scores(q.detach(), bound_min, bound_max)
        selected = select_pages(
            scores, page_lens, num_pages, query_positions, page_size, k_pages_k, n_recent
        )
    if selected.shape[-1] == 0:
        return torch.zeros_like(query), selected

    safe = selected.clamp(min=0)
    num_slots = k_vals.shape[2]
    page_size_actual = k_vals.shape[3]
    gathered_index = safe[:, :, :, :, None, None].expand(
        batch, num_q_heads, queries, selected.shape[-1], page_size_actual, k_vals.shape[-1]
    )
    k_src = k_vals.unsqueeze(2).expand(batch, num_q_heads, queries, num_slots, page_size_actual, -1)
    v_src = v_vals.unsqueeze(2).expand(batch, num_q_heads, queries, num_slots, page_size_actual, -1)
    k_sel = torch.gather(k_src, 3, gathered_index)
    v_sel = torch.gather(v_src, 3, gathered_index)
    valid_page = selected >= 0
    k_sel = k_sel * valid_page[:, :, :, :, None, None].to(k_sel.dtype)
    v_sel = v_sel * valid_page[:, :, :, :, None, None].to(v_sel.dtype)

    local = torch.arange(page_size_actual, device=device)
    token_pos = safe.unsqueeze(-1) * page_size + local.view(1, 1, 1, 1, page_size_actual)
    lens_src = page_lens[:, None, None, :].expand(batch, num_q_heads, queries, num_slots)
    gathered_lens = torch.gather(lens_src, -1, safe)
    in_page = local.view(1, 1, 1, 1, page_size_actual) < gathered_lens.unsqueeze(-1)
    causal = token_pos <= query_positions.view(batch, 1, queries, 1, 1)
    token_ok = valid_page.unsqueeze(-1) & in_page & causal

    if approx:
        page_logit = torch.gather(scores, -1, safe)
        logits = page_logit.unsqueeze(-1).expand_as(token_ok).float() * scale
    else:
        logits = torch.einsum("bhtd,bhtksd->bhtks", q, k_sel) * scale
    logits = logits.masked_fill(~token_ok, float("-inf"))
    flat = logits.reshape(*logits.shape[:-2], -1)
    weights = torch.softmax(flat, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    if prune_threshold > 0:
        weights = torch.where(weights >= prune_threshold, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    weights = weights.view_as(logits)
    out = torch.einsum("bhtks,bhtksd->bhtd", weights, v_sel)
    return out.to(dtype=out_dtype), selected


def sparse_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    page_size: int,
    k_pages: int,
    n_recent: int,
    scale: float,
    query_positions: Optional[torch.Tensor] = None,
    lengths: Optional[torch.Tensor] = None,
    prune_threshold: float = 0.0,
    approx: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Query-aware attention over a dense sequence, used by training."""
    batch, _, queries, _ = query.shape
    tokens = key.shape[2]
    if lengths is None:
        lengths = torch.full((batch,), tokens, device=key.device, dtype=torch.long)
    if query_positions is None:
        if queries != tokens:
            raise ValueError("query_positions is required when query and key lengths differ")
        query_positions = torch.arange(queries, device=query.device).view(1, queries).expand(batch, queries)
    k_pages_t, k_lens, num_pages = paginate(key, page_size, lengths)
    v_pages_t, _, _ = paginate(value, page_size, lengths)
    with torch.no_grad():
        k_min, k_max = page_bounds(k_pages_t.detach(), k_lens)
    return sparse_cached_attention(
        query,
        k_pages_t,
        v_pages_t,
        k_min,
        k_max,
        k_lens,
        num_pages,
        page_size=page_size,
        k_pages_k=k_pages,
        n_recent=n_recent,
        scale=scale,
        query_positions=query_positions,
        prune_threshold=prune_threshold,
        approx=approx,
    )


def dense_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Full causal attention. Query and key positions both start at zero."""
    out_dtype = query.dtype
    q = query.float()
    k = expand_kv(key.float(), q.shape[1])
    v = expand_kv(value.float(), q.shape[1])
    logits = torch.einsum("bhtd,bhsd->bhts", q, k) * scale
    q_pos = torch.arange(q.shape[2], device=q.device)
    k_pos = torch.arange(k.shape[2], device=q.device)
    causal = k_pos.view(1, 1, 1, -1) <= q_pos.view(1, 1, -1, 1)
    logits = logits.masked_fill(~causal, float("-inf"))
    weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
    return torch.einsum("bhts,bhsd->bhtd", weights, v).to(dtype=out_dtype)
