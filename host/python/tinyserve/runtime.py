"""Decode loop, entropy early-exit, and session placement."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Dict, Iterator, List, Optional, Sequence

import torch

from .cache import PagedKVCache
from .model import TinyServeLM


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of a next-token distribution, in nats. ``logits`` is ``[B, V]``."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def sample_tokens(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Greedy when ``temperature`` is 0, otherwise multinomial."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    scaled = logits.float() / temperature
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def page_overlap(previous: torch.Tensor, current: torch.Tensor) -> float:
    """Fraction of current page ids that were also selected on the previous step."""
    if previous is None or current is None or current.numel() == 0:
        return 0.0
    valid = current >= 0
    if int(valid.sum()) == 0:
        return 0.0
    hit = (current.unsqueeze(-1) == previous.unsqueeze(-2)).any(dim=-1) & valid
    return float((hit.sum().float() / valid.sum()).item())


@dataclass
class Profiler:
    """Layer-agnostic wall-clock spans. CUDA events are used when a tensor is supplied."""

    records: List[tuple] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, tensor: Optional[torch.Tensor] = None) -> Iterator[None]:
        use_cuda = tensor is not None and tensor.is_cuda and torch.cuda.is_available()
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                end.synchronize()
                self.records.append((name, float(start.elapsed_time(end))))
        else:
            start_s = time.perf_counter()
            try:
                yield
            finally:
                self.records.append((name, (time.perf_counter() - start_s) * 1e3))

    def summary(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for name, millis in self.records:
            totals[name] = totals.get(name, 0.0) + millis
            counts[name] = counts.get(name, 0) + 1
        return {name: totals[name] / counts[name] for name in totals}


@dataclass
class GenerateResult:
    tokens: torch.Tensor
    lengths: torch.Tensor
    page_hit_rate: float
    profile: Dict[str, float]
    stopped_early: bool


class ServingScheduler:
    """Place sessions on the least-loaded device and migrate when imbalance is large.

    This is the session-affinity policy behind the paper's multi-GPU serving
    measurements: a request stays where its KV already resides unless moving
    a small session reduces the load gap.
    """

    def __init__(self, n_devices: int = 1) -> None:
        if n_devices < 1:
            raise ValueError("n_devices must be positive")
        self.n_devices = n_devices
        self.load = [0] * n_devices
        self.home: Dict[int, int] = {}
        self.size: Dict[int, int] = {}

    def admit(self, session_id: int, n_tokens: int) -> int:
        if session_id in self.home:
            raise ValueError(f"session {session_id} is already admitted")
        device = min(range(self.n_devices), key=lambda index: (self.load[index], index))
        self.home[session_id] = device
        self.size[session_id] = n_tokens
        self.load[device] += n_tokens
        return device

    def imbalance(self) -> float:
        if self.n_devices == 1 or max(self.load) == 0:
            return 0.0
        return (max(self.load) - min(self.load)) / max(self.load)

    def migrate(self, threshold: float = 0.25) -> List[tuple]:
        """Move the smallest session off the hottest device when that reduces imbalance."""
        if self.n_devices < 2 or not self.home:
            return []
        src = max(range(self.n_devices), key=lambda index: (self.load[index], -index))
        dst = min(range(self.n_devices), key=lambda index: (self.load[index], index))
        if self.load[src] == 0 or self.imbalance() < threshold:
            return []
        candidates = [sid for sid, device in self.home.items() if device == src]
        if not candidates:
            return []
        session_id = min(candidates, key=lambda sid: (self.size[sid], sid))
        gap = self.load[src] - self.load[dst]
        if self.size[session_id] >= gap:
            return []
        self.home[session_id] = dst
        self.load[src] -= self.size[session_id]
        self.load[dst] += self.size[session_id]
        return [(session_id, src, dst)]


def _generate(model, prompt, max_new_tokens, temperature, profiler) -> GenerateResult:
    config = model.config
    batch, prompt_len = prompt.shape
    profiler = profiler or Profiler()
    cache = PagedKVCache.create(config, batch=batch, device=prompt.device)
    with profiler.span("prefill", prompt):
        logits = model.prefill(prompt, cache)
    next_logits = logits[:, -1, :]
    generated = prompt.new_zeros(batch, max_new_tokens)
    stop_at = torch.full((batch,), max_new_tokens, dtype=torch.long)
    stopped = torch.zeros(batch, dtype=torch.bool)
    previous: Dict[int, torch.Tensor] = {}
    hit_values: List[float] = []

    for step in range(max_new_tokens):
        if config.entropy_threshold is not None:
            newly = (~stopped) & (token_entropy(next_logits).cpu() < config.entropy_threshold)
            stop_at = torch.where(newly, torch.full_like(stop_at, step), stop_at)
            stopped = stopped | newly
            if bool(stopped.all()):
                break
        nxt = sample_tokens(next_logits, temperature)
        with profiler.span("decode", next_logits):
            step_logits = model.decode_token(nxt, cache)
        generated[:, step] = nxt
        next_logits = step_logits[:, -1, :]
        for layer, selected in cache.last_selected.items():
            current = selected[:, :, 0, :] if selected.dim() == 4 else selected
            if layer in previous and previous[layer].shape == current.shape:
                hit_values.append(page_overlap(previous[layer], current))
            previous[layer] = current.detach()

    lengths = prompt_len + stop_at
    hit = sum(hit_values) / len(hit_values) if hit_values else 0.0
    return GenerateResult(
        tokens=torch.cat([prompt, generated], dim=1),
        lengths=lengths,
        page_hit_rate=hit,
        profile=profiler.summary(),
        stopped_early=bool(stopped.any()),
    )


@torch.no_grad()
def generate(
    model: TinyServeLM,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    profiler: Optional[Profiler] = None,
) -> GenerateResult:
    """Prefill with dense attention, then decode with query-aware page selection.

    Entropy early-exit uses ``model.config.entropy_threshold`` in nats. A
    sequence that falls below the threshold stops; the batch stays lockstep
    until every sequence has stopped, and ``lengths`` records the valid prefix.
    """
    if prompt.dim() != 2:
        raise ValueError("prompt must have shape [batch, tokens]")
    was_training = model.training
    model.eval()
    try:
        return _generate(model, prompt, max_new_tokens, temperature, profiler)
    finally:
        model.train(was_training)


def dispatch_batch(fn, batch: torch.Tensor, devices: Sequence[torch.device]) -> torch.Tensor:
    """Run ``fn`` on batch shards and concatenate on the input device.

    With one device this is a plain call. With several devices the batch is
    split along dim 0, which is the data-parallel dispatch used for concurrent
    prefill.
    """
    if len(devices) == 0:
        raise ValueError("devices must be non-empty")
    if len(devices) == 1:
        return fn(batch.to(devices[0])).to(batch.device)
    if batch.shape[0] < len(devices):
        raise ValueError("batch is smaller than the number of devices")
    pieces = []
    for chunk, device in zip(batch.chunk(len(devices), dim=0), devices):
        pieces.append(fn(chunk.to(device)).to(batch.device))
    return torch.cat(pieces, dim=0)
