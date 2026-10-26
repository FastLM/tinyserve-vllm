"""Runtime configuration for query-aware page selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class TinyServeConfig:
    """Decoder and cache settings.

    ``top_k_ratio`` is the fraction of KV pages scored into the decode budget.
    ``token_budget`` caps that fraction at ``budget // page_size`` pages, which
    is the fixed token budget used in the paper's serving experiments.
    ``n_recent`` pages ending at the current token are always resident so the
    newest keys cannot be dropped by the bounding-box estimate.
    """

    n_layer: int = 2
    n_embd: int = 64
    n_head: int = 4
    n_kv_head: int = 4
    vocab_size: int = 128
    block_size: int = 256
    page_size: int = 16
    top_k_ratio: float = 0.3
    token_budget: Optional[int] = None
    n_recent: int = 1
    dropout: float = 0.0
    prune_threshold: float = 0.0
    approx_attention: bool = False
    kv_storage: str = "fp16"
    checkpoint: bool = False
    dense_train: bool = False
    entropy_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.n_recent < 0:
            raise ValueError("n_recent must be non-negative")
        if not 0 < self.top_k_ratio <= 1:
            raise ValueError("top_k_ratio must be in (0, 1]")
        if self.kv_storage not in ("fp32", "fp16", "int8"):
            raise ValueError("kv_storage must be fp32, fp16, or int8")
        self.head_dim = self.n_embd // self.n_head

    @property
    def max_pages(self) -> int:
        return math.ceil(self.block_size / self.page_size)


def resolve_k(
    num_pages: int,
    page_size: int,
    top_k_ratio: float,
    token_budget: Optional[int],
) -> int:
    """Number of pages fetched at a decode step."""
    if num_pages <= 0:
        return 0
    k = max(1, int(round(top_k_ratio * num_pages)))
    k = min(k, num_pages)
    if token_budget is not None:
        k = min(k, max(1, int(token_budget) // int(page_size)))
    return max(1, min(num_pages, k))


def preset(name: str) -> TinyServeConfig:
    """Architecture shapes in the same class as the paper's serving models.

    These are randomly initialized decoders for the serving stack. They are
    not the pretrained TinyLLaMA / GPT-2 / OPT / LLaMA checkpoints.
    """
    table = {
        "debug": TinyServeConfig(
            n_layer=2,
            n_embd=64,
            n_head=4,
            n_kv_head=2,
            vocab_size=128,
            block_size=128,
            page_size=8,
            top_k_ratio=0.5,
            kv_storage="fp32",
        ),
        "tinyllama-125m": TinyServeConfig(
            n_layer=12,
            n_embd=768,
            n_head=12,
            n_kv_head=12,
            vocab_size=32000,
            block_size=2048,
            page_size=16,
            top_k_ratio=0.3,
            token_budget=2048,
        ),
        "gpt2-345m": TinyServeConfig(
            n_layer=24,
            n_embd=1024,
            n_head=16,
            n_kv_head=16,
            vocab_size=50257,
            block_size=1024,
            page_size=16,
            top_k_ratio=0.3,
            token_budget=2048,
        ),
        "opt-350m": TinyServeConfig(
            n_layer=24,
            n_embd=1024,
            n_head=16,
            n_kv_head=16,
            vocab_size=50272,
            block_size=2048,
            page_size=16,
            top_k_ratio=0.3,
            token_budget=2048,
        ),
        "gpt2-774m": TinyServeConfig(
            n_layer=36,
            n_embd=1280,
            n_head=20,
            n_kv_head=20,
            vocab_size=50257,
            block_size=1024,
            page_size=16,
            top_k_ratio=0.3,
            token_budget=2048,
        ),
        "llama-1.3b": TinyServeConfig(
            n_layer=24,
            n_embd=2048,
            n_head=16,
            n_kv_head=4,
            vocab_size=32000,
            block_size=2048,
            page_size=16,
            top_k_ratio=0.3,
            token_budget=2048,
        ),
    }
    try:
        return table[name]
    except KeyError as exc:
        known = ", ".join(sorted(table))
        raise KeyError(f"unknown preset {name!r}; expected one of {known}") from exc


PRESETS = (
    "debug",
    "tinyllama-125m",
    "gpt2-345m",
    "opt-350m",
    "gpt2-774m",
    "llama-1.3b",
)
