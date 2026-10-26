"""Paged KV cache with incremental bounding-box metadata.

Pages hold ``page_size`` tokens. fp16 and fp32 pages store the keys directly.
int8 pages keep a one-page fp32 shadow, requantize that page on every append,
and record the scale the attention kernel uses to dequantize. Bounds are
always fp32 and describe the values attention will actually read.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .attention import page_bounds, paginate, sparse_cached_attention
from .config import TinyServeConfig, resolve_k


class PagedKVCache:
    def __init__(
        self,
        *,
        n_layers: int,
        batch: int,
        num_kv_heads: int,
        page_size: int,
        head_dim: int,
        max_pages: int,
        max_tokens: int,
        storage: str,
        device: torch.device,
    ) -> None:
        if storage not in ("fp32", "fp16", "int8"):
            raise ValueError(f"unsupported storage {storage}")
        self.n_layers = n_layers
        self.batch = batch
        self.num_kv_heads = num_kv_heads
        self.page_size = page_size
        self.head_dim = head_dim
        self.max_pages = max_pages
        self.max_tokens = max_tokens
        self.storage = storage
        self.device = torch.device(device)
        shape = (n_layers, batch, num_kv_heads, max_pages, page_size, head_dim)
        bound = (n_layers, batch, num_kv_heads, max_pages, head_dim)
        self.kmin = torch.zeros(*bound, dtype=torch.float32, device=self.device)
        self.kmax = torch.zeros(*bound, dtype=torch.float32, device=self.device)
        self.page_lens = torch.zeros(n_layers, batch, max_pages, dtype=torch.long, device=self.device)
        self.num_pages = torch.zeros(n_layers, batch, dtype=torch.long, device=self.device)
        self.seq_len = torch.zeros(batch, dtype=torch.long)
        self.last_selected: Dict[int, torch.Tensor] = {}
        if storage == "int8":
            self.k_i8 = torch.zeros(*shape, dtype=torch.int8, device=self.device)
            self.v_i8 = torch.zeros(*shape, dtype=torch.int8, device=self.device)
            scale_shape = (n_layers, batch, num_kv_heads, max_pages)
            self.k_scale = torch.ones(*scale_shape, dtype=torch.float32, device=self.device)
            self.v_scale = torch.ones(*scale_shape, dtype=torch.float32, device=self.device)
            shadow = (n_layers, batch, num_kv_heads, page_size, head_dim)
            self.shadow_k = torch.zeros(*shadow, dtype=torch.float32, device=self.device)
            self.shadow_v = torch.zeros(*shadow, dtype=torch.float32, device=self.device)
            self.k = None
            self.v = None
        else:
            dtype = torch.float32 if storage == "fp32" else torch.float16
            self.k = torch.zeros(*shape, dtype=dtype, device=self.device)
            self.v = torch.zeros(*shape, dtype=dtype, device=self.device)
            self.k_i8 = self.v_i8 = None
            self.k_scale = self.v_scale = None
            self.shadow_k = self.shadow_v = None

    @classmethod
    def create(cls, config: TinyServeConfig, batch: int, device: torch.device) -> "PagedKVCache":
        return cls(
            n_layers=config.n_layer,
            batch=batch,
            num_kv_heads=config.n_kv_head,
            page_size=config.page_size,
            head_dim=config.head_dim,
            max_pages=config.max_pages,
            max_tokens=config.block_size,
            storage=config.kv_storage,
            device=device,
        )

    def advance(self) -> None:
        self.seq_len += 1

    def write_sequence(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Replace one layer's cache with the keys and values of a prefill."""
        if key.shape[0] != self.batch:
            raise ValueError("prefill batch does not match the cache")
        tokens = key.shape[2]
        if tokens > self.max_tokens:
            raise RuntimeError("prefill exceeds block_size")
        lengths = torch.full((self.batch,), tokens, dtype=torch.long, device=key.device)
        self.page_lens[layer].zero_()
        self.num_pages[layer].zero_()
        self.kmin[layer].zero_()
        self.kmax[layer].zero_()
        if self.storage == "int8":
            self.k_i8[layer].zero_()
            self.v_i8[layer].zero_()
            self._write_int8(layer, key.float(), value.float(), lengths)
        else:
            assert self.k is not None and self.v is not None
            self.k[layer].zero_()
            self.v[layer].zero_()
            stored_k = key.to(dtype=self.k.dtype)
            stored_v = value.to(dtype=self.v.dtype)
            pages_k, lens, num_pages = paginate(stored_k, self.page_size, lengths)
            pages_v, _, _ = paginate(stored_v, self.page_size, lengths)
            slots = pages_k.shape[2]
            self.k[layer, :, :, :slots].copy_(pages_k)
            self.v[layer, :, :, :slots].copy_(pages_v)
            lower, upper = page_bounds(pages_k.float(), lens)
            self.kmin[layer, :, :, :slots].copy_(lower)
            self.kmax[layer, :, :, :slots].copy_(upper)
            self.page_lens[layer, :, :slots].copy_(lens)
            self.num_pages[layer].copy_(num_pages)
        self.seq_len[:] = tokens

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write one new token at ``seq_len`` without incrementing it.

        Every layer in a decode step observes the same position. The caller
        advances the cache after the last layer.
        """
        if not torch.all(self.seq_len == self.seq_len[0]):
            raise RuntimeError("TinyServe cache expects lockstep batch lengths")
        pos = int(self.seq_len[0].item())
        if pos >= self.max_tokens:
            raise RuntimeError("decode exceeds block_size")
        page = pos // self.page_size
        slot = pos % self.page_size
        if self.storage == "int8":
            self._append_int8(layer, key.float(), value.float(), page, slot)
        else:
            assert self.k is not None and self.v is not None
            stored_k = key.to(dtype=self.k.dtype)
            stored_v = value.to(dtype=self.v.dtype)
            self.k[layer, :, :, page, slot, :].copy_(stored_k)
            self.v[layer, :, :, page, slot, :].copy_(stored_v)
            as_float = stored_k.float()
            if slot == 0:
                self.kmin[layer, :, :, page, :].copy_(as_float)
                self.kmax[layer, :, :, page, :].copy_(as_float)
            else:
                bound_min = self.kmin[layer, :, :, page, :]
                bound_max = self.kmax[layer, :, :, page, :]
                self.kmin[layer, :, :, page, :].copy_(torch.minimum(bound_min, as_float))
                self.kmax[layer, :, :, page, :].copy_(torch.maximum(bound_max, as_float))
        self.page_lens[layer, :, page] = slot + 1
        self.num_pages[layer, :] = page + 1

    def attend(
        self,
        layer: int,
        query: torch.Tensor,
        positions: torch.Tensor,
        config: TinyServeConfig,
        scale: float,
    ) -> torch.Tensor:
        if query.dim() == 3:
            query = query.unsqueeze(2)
        batch = query.shape[0]
        qp = positions.to(device=query.device, dtype=torch.long).view(batch, -1)
        pages = int(self.num_pages[layer].max().item())
        k_sel = resolve_k(pages, self.page_size, config.top_k_ratio, config.token_budget)
        if self.storage == "int8":
            key, value = self.k_i8[layer], self.v_i8[layer]
            k_scale, v_scale = self.k_scale[layer], self.v_scale[layer]
        else:
            assert self.k is not None and self.v is not None
            key, value = self.k[layer], self.v[layer]
            k_scale = v_scale = None
        out, selected = sparse_cached_attention(
            query,
            key,
            value,
            self.kmin[layer],
            self.kmax[layer],
            self.page_lens[layer],
            self.num_pages[layer],
            page_size=self.page_size,
            k_pages_k=k_sel,
            n_recent=config.n_recent,
            scale=scale,
            query_positions=qp,
            prune_threshold=config.prune_threshold,
            approx=config.approx_attention,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        self.last_selected[layer] = selected.detach()
        return out

    def _write_int8(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        pages_k, lens, num_pages = paginate(key, self.page_size, lengths)
        pages_v, _, _ = paginate(value, self.page_size, lengths)
        slots = pages_k.shape[2]
        local = torch.arange(self.page_size, device=key.device)
        valid = local.view(1, 1, 1, self.page_size) < lens.view(-1, 1, slots, 1)
        valid = valid.unsqueeze(-1)
        masked_k = pages_k.masked_fill(~valid, 0)
        masked_v = pages_v.masked_fill(~valid, 0)
        k_scale = masked_k.abs().amax(dim=(-1, -2)).clamp_min(1e-8) / 127.0
        v_scale = masked_v.abs().amax(dim=(-1, -2)).clamp_min(1e-8) / 127.0
        qk = torch.clamp(torch.round(masked_k / k_scale[:, :, :, None, None]), -127, 127).to(torch.int8)
        qv = torch.clamp(torch.round(masked_v / v_scale[:, :, :, None, None]), -127, 127).to(torch.int8)
        self.k_i8[layer, :, :, :slots].copy_(qk)
        self.v_i8[layer, :, :, :slots].copy_(qv)
        self.k_scale[layer, :, :, :slots].copy_(k_scale)
        self.v_scale[layer, :, :, :slots].copy_(v_scale)
        dequant = qk.float() * k_scale[:, :, :, None, None]
        lower, upper = page_bounds(dequant, lens)
        self.kmin[layer, :, :, :slots].copy_(lower)
        self.kmax[layer, :, :, :slots].copy_(upper)
        self.page_lens[layer, :, :slots].copy_(lens.to(self.page_lens.device))
        self.num_pages[layer].copy_(num_pages.to(self.num_pages.device))

    def _append_int8(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        page: int,
        slot: int,
    ) -> None:
        assert self.shadow_k is not None and self.shadow_v is not None
        assert self.k_i8 is not None and self.v_i8 is not None
        assert self.k_scale is not None and self.v_scale is not None
        if slot == 0:
            self.shadow_k[layer].zero_()
            self.shadow_v[layer].zero_()
            self.k_i8[layer, :, :, page].zero_()
            self.v_i8[layer, :, :, page].zero_()
        self.shadow_k[layer, :, :, slot, :].copy_(key)
        self.shadow_v[layer, :, :, slot, :].copy_(value)
        active_k = self.shadow_k[layer, :, :, : slot + 1, :]
        active_v = self.shadow_v[layer, :, :, : slot + 1, :]
        k_scale = active_k.abs().amax(dim=(-1, -2)).clamp_min(1e-8) / 127.0
        v_scale = active_v.abs().amax(dim=(-1, -2)).clamp_min(1e-8) / 127.0
        qk = torch.clamp(torch.round(active_k / k_scale[:, :, None, None]), -127, 127).to(torch.int8)
        qv = torch.clamp(torch.round(active_v / v_scale[:, :, None, None]), -127, 127).to(torch.int8)
        self.k_i8[layer, :, :, page, : slot + 1].copy_(qk)
        self.v_i8[layer, :, :, page, : slot + 1].copy_(qv)
        self.k_scale[layer, :, :, page].copy_(k_scale)
        self.v_scale[layer, :, :, page].copy_(v_scale)
        dequant = qk.float() * k_scale[:, :, None, None]
        self.kmin[layer, :, :, page, :].copy_(dequant.amin(dim=2))
        self.kmax[layer, :, :, page, :].copy_(dequant.amax(dim=2))
