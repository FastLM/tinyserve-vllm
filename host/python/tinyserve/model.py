"""Small decoder that runs TinyServe's prefill and decode paths."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import dense_causal_attention, sparse_causal_attention
from .cache import PagedKVCache
from .config import TinyServeConfig, resolve_k


class SelfAttention(nn.Module):
    def __init__(self, config: TinyServeConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv = config.n_kv_head
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

    def project(self, hidden: torch.Tensor):
        batch, tokens, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, tokens, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        key = self.k_proj(hidden).view(batch, tokens, self.n_kv, self.head_dim).permute(0, 2, 1, 3)
        value = self.v_proj(hidden).view(batch, tokens, self.n_kv, self.head_dim).permute(0, 2, 1, 3)
        return query, key, value

    def combine(self, context: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, dim = context.shape
        merged = context.permute(0, 2, 1, 3).contiguous().view(batch, tokens, heads * dim)
        return self.o_proj(merged)


class MLP(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(hidden)))


class DecoderBlock(nn.Module):
    def __init__(self, config: TinyServeConfig) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward_tokens(
        self,
        hidden: torch.Tensor,
        *,
        dense: bool,
        cache: Optional[PagedKVCache] = None,
        layer_idx: int = 0,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden
        query, key, value = self.attn.project(self.ln1(hidden))
        scale = self.attn.scale
        cfg = self.config
        if cache is None:
            if dense:
                context = dense_causal_attention(query, key, value, scale)
            else:
                num_pages = math.ceil(key.shape[2] / cfg.page_size)
                context, _ = sparse_causal_attention(
                    query,
                    key,
                    value,
                    page_size=cfg.page_size,
                    k_pages=resolve_k(num_pages, cfg.page_size, cfg.top_k_ratio, cfg.token_budget),
                    n_recent=cfg.n_recent,
                    scale=scale,
                    prune_threshold=cfg.prune_threshold,
                    approx=cfg.approx_attention,
                )
        elif hidden.shape[1] > 1:
            context = dense_causal_attention(query, key, value, scale)
            cache.write_sequence(layer_idx, key, value)
        else:
            if positions is None:
                raise ValueError("decode requires token positions")
            cache.append(layer_idx, key.squeeze(2), value.squeeze(2))
            context = cache.attend(layer_idx, query, positions, cfg, scale)
        hidden = residual + self.drop(self.attn.combine(context))
        hidden = hidden + self.drop(self.mlp(self.ln2(hidden)))
        return hidden


class TinyServeLM(nn.Module):
    def __init__(self, config: TinyServeConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed(self, token_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden = self.wte(token_ids)
        if positions.dim() == 1 and positions.shape[0] == token_ids.shape[1]:
            positional = self.wpe(positions).unsqueeze(0)
        elif positions.dim() == 1 and positions.shape[0] == token_ids.shape[0]:
            positional = self.wpe(positions).unsqueeze(1)
        else:
            positional = self.wpe(positions)
        return hidden + positional

    def _run_blocks(
        self,
        hidden: torch.Tensor,
        *,
        dense: bool,
        cache: Optional[PagedKVCache],
        positions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for index, block in enumerate(self.blocks):
            if self.training and self.config.checkpoint and cache is None:
                def rerun(tokens, block=block, dense=dense):
                    return block.forward_tokens(tokens, dense=dense)

                hidden = torch.utils.checkpoint.checkpoint(rerun, hidden, use_reentrant=False)
            else:
                hidden = block.forward_tokens(
                    hidden,
                    dense=dense,
                    cache=cache,
                    layer_idx=index,
                    positions=positions,
                )
        return self.lm_head(self.ln_f(hidden))

    def forward(self, token_ids: torch.Tensor, dense: Optional[bool] = None) -> torch.Tensor:
        """Teacher-forced logits. ``dense=False`` uses query-aware sparsity."""
        if dense is None:
            dense = self.config.dense_train
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        return self._run_blocks(self.embed(token_ids, positions), dense=dense, cache=None, positions=None)

    def prefill(self, token_ids: torch.Tensor, cache: PagedKVCache) -> torch.Tensor:
        """Exact causal prefill that materializes paged KV and bounds."""
        if token_ids.shape[1] > self.config.block_size:
            raise RuntimeError("prefill exceeds block_size")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        return self._run_blocks(self.embed(token_ids, positions), dense=True, cache=cache, positions=None)

    def decode_token(self, token_ids: torch.Tensor, cache: PagedKVCache) -> torch.Tensor:
        """One query-aware decode step. ``token_ids`` is ``[B]`` or ``[B, 1]``."""
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(1)
        if token_ids.shape[1] != 1:
            raise ValueError("decode_token expects a single new token")
        positions = cache.seq_len.to(device=token_ids.device)
        if int(positions.max()) >= self.config.block_size:
            raise RuntimeError("decode exceeds block_size")
        logits = self._run_blocks(
            self.embed(token_ids, positions),
            dense=False,
            cache=cache,
            positions=positions,
        )
        cache.advance()
        return logits
