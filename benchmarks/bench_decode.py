"""Microbenchmark dense decode against query-aware sparse decode.

This does not reproduce the paper's end-to-end speedups. It times the
attention math on one decode token and prints the traffic fraction from the
cost model.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host" / "python"))

import torch

from tinyserve.attention import expand_kv, sparse_causal_attention
from tinyserve.config import resolve_k
from tinyserve.metrics import memory_fraction


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1e3 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query = torch.randn(args.batch, args.heads, 1, args.dim, device=device)
    key = torch.randn(args.batch, args.kv_heads, args.tokens, args.dim, device=device)
    value = torch.randn(args.batch, args.kv_heads, args.tokens, args.dim, device=device)
    scale = args.dim ** -0.5
    pages = (args.tokens + args.page_size - 1) // args.page_size
    k_pages = resolve_k(pages, args.page_size, args.ratio, None)
    positions = torch.full((args.batch, 1), args.tokens - 1, device=device)

    def dense_tail():
        # The last token may attend to the whole cache. dense_causal_attention
        # assumes the query positions are 0..Tq-1, so it is the wrong mask here.
        q = query.float()
        k = expand_kv(key.float(), q.shape[1])
        v = expand_kv(value.float(), q.shape[1])
        weights = torch.softmax(torch.einsum("bhtd,bhsd->bhts", q, k) * scale, dim=-1)
        return torch.einsum("bhts,bhsd->bhtd", weights, v)

    dense_ms = bench(dense_tail, args.iters)
    sparse_ms = bench(
        lambda: sparse_causal_attention(
            query,
            key,
            value,
            page_size=args.page_size,
            k_pages=k_pages,
            n_recent=1,
            scale=scale,
            query_positions=positions,
        ),
        args.iters,
    )
    fraction = memory_fraction(args.tokens, args.page_size, k_pages, reuse=1.0)
    print(f"device={device} tokens={args.tokens} pages={pages} selected={k_pages}")
    print(f"dense={dense_ms:.3f} ms  sparse={sparse_ms:.3f} ms  ratio={dense_ms / sparse_ms:.2f}x")
    print(f"memory fraction at reuse=1: {fraction:.3f}")


if __name__ == "__main__":
    main()
