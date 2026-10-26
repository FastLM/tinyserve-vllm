"""Streaming top-k that the fused kernel implements.

The CUDA kernel scores pages in tiles of 128, bitonic-sorts each tile, and
merges the tile into a sorted top-k. Ties break toward the smaller page id.
After selection, the most recent pages are pinned so decode always sees the
token that was just written.

The PyTorch path ranks finite recent pages ahead of the rest, newest first,
then fills the budget by score. ``bonus_select`` is that policy. It returns
the same page set as streaming top-k followed by ``pin_recent``.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

NEG_INF = float("-inf")
Pair = Tuple[float, int]


def better(left: Pair, right: Pair) -> bool:
    """True when ``left`` should rank above ``right``."""
    score_l, id_l = left
    score_r, id_r = right
    if score_l != score_r:
        return score_l > score_r
    return id_l < id_r


def bitonic_desc(values: Sequence[Pair]) -> List[Pair]:
    """Descending bitonic sort. ``len(values)`` must be a power of two."""
    vals = list(values)
    n = len(vals)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("bitonic sort expects a non-empty power-of-two length")
    span = 2
    while span <= n:
        stride = span >> 1
        while stride > 0:
            for i in range(n):
                partner = i ^ stride
                if partner > i:
                    left_better = better(vals[i], vals[partner])
                    if (i & span) == 0:
                        if not left_better:
                            vals[i], vals[partner] = vals[partner], vals[i]
                    elif left_better:
                        vals[i], vals[partner] = vals[partner], vals[i]
            stride >>= 1
        span <<= 1
    return vals


def merge_topk(top: Sequence[Pair], tile: Sequence[Pair], k: int) -> List[Pair]:
    """Merge two descending lists and keep ``k`` elements."""
    i = 0
    j = 0
    merged: List[Pair] = []
    while len(merged) < k:
        if i >= len(top):
            merged.append(tile[j])
            j += 1
        elif j >= len(tile):
            merged.append(top[i])
            i += 1
        elif better(top[i], tile[j]):
            merged.append(top[i])
            i += 1
        elif better(tile[j], top[i]):
            merged.append(tile[j])
            j += 1
        else:
            merged.append(top[i])
            i += 1
    return merged


def streaming_topk(
    scores: Sequence[float],
    ids: Sequence[int],
    k: int,
    tile: int = 128,
) -> List[Pair]:
    """Tile-wise bitonic top-k. Returns ``k`` ``(score, id)`` pairs, best first."""
    if tile <= 0 or (tile & (tile - 1)) != 0:
        raise ValueError("tile must be a positive power of two")
    if k <= 0:
        return []
    if len(scores) != len(ids):
        raise ValueError("scores and ids must have the same length")
    n = len(scores)
    pad = (tile - (n % tile)) % tile
    pairs: List[Pair] = list(zip(scores, ids)) + [(NEG_INF, -1)] * pad
    top = [(NEG_INF, -1)] * k
    for base in range(0, len(pairs), tile):
        chunk = bitonic_desc(pairs[base : base + tile])
        top = merge_topk(top, chunk, k)
    return top


def pin_recent(selected: Sequence[int], num_pages: int, n_recent: int) -> List[int]:
    """Force the last ``n_recent`` pages into a best-first selection."""
    chosen = list(selected)
    k = len(chosen)
    if num_pages <= 0 or n_recent <= 0 or k == 0:
        return chosen
    recent_lo = max(0, num_pages - n_recent)
    recent = list(range(recent_lo, num_pages))
    present = {page for page in chosen if page >= 0}
    missing = [page for page in recent if page not in present]
    if not missing:
        return chosen
    must = set(recent)
    evictable = [i for i in range(k - 1, -1, -1) if chosen[i] not in must]
    if len(missing) > len(evictable):
        start = max(0, num_pages - k)
        out = list(range(start, num_pages))
        out.extend([-1] * (k - len(out)))
        return out[:k]
    for page, slot in zip(missing, evictable):
        chosen[slot] = page
    return chosen


def bonus_select(scores: Sequence[float], k: int, n_recent: int) -> List[int]:
    """Lexicographic select used by the PyTorch reference.

    Finite recent pages sort ahead of every other page, newest first. Inside
    that group, and among the remaining pages, the higher score wins and ties
    break toward the smaller page id.
    """
    n = len(scores)
    if k <= 0 or n == 0:
        return []
    recent_lo = max(0, n - n_recent) if n_recent > 0 else n

    def sort_key(page: int):
        score = scores[page]
        finite = score == score and score != NEG_INF
        if page >= recent_lo and finite:
            age = n - 1 - page
            priority = n_recent - age
        else:
            priority = 0
        return (-priority, -score, page)

    return sorted(range(n), key=sort_key)[: min(k, n)]


def streaming_select(
    scores: Sequence[float],
    k: int,
    n_recent: int,
    tile: int = 128,
) -> List[int]:
    """Kernel policy: streaming top-k, then pin the newest pages."""
    ids = list(range(len(scores)))
    top = streaming_topk(scores, ids, k, tile=tile)
    return pin_recent([ident for _, ident in top], len(scores), n_recent)


def same_page_set(left: Iterable[int], right: Iterable[int]) -> bool:
    return {page for page in left if page >= 0} == {page for page in right if page >= 0}
