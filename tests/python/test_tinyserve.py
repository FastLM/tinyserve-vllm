import math
import unittest

import torch
import torch.nn.functional as F

from tinyserve.attention import (
    bounding_box_scores,
    dense_causal_attention,
    page_bounds,
    sparse_causal_attention,
)
from tinyserve.cache import PagedKVCache
from tinyserve.config import TinyServeConfig, preset, resolve_k
from tinyserve.metrics import (
    approximation_gap_bound,
    fraction_at_paper_page_size,
    memory_fraction,
    memory_load,
    minimized_memory_fraction,
    minimizing_page_size,
    paper_page_size,
)
from tinyserve.model import TinyServeLM
from tinyserve.runtime import (
    ServingScheduler,
    dispatch_batch,
    generate,
    page_overlap,
    token_entropy,
)
from tinyserve.train import AsyncGradSync, page_importance, retain_mask


def _tiny_config(**overrides) -> TinyServeConfig:
    values = dict(
        n_layer=2,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        vocab_size=50,
        block_size=64,
        page_size=4,
        top_k_ratio=1.0,
        token_budget=None,
        n_recent=1,
        kv_storage="fp32",
        dropout=0.0,
    )
    values.update(overrides)
    return TinyServeConfig(**values)


class MetricTest(unittest.TestCase):
    def test_fraction_matches_the_load_formula(self):
        length, page, pages, reuse, width = 32768, 16, 128, 0.5, 4.0
        fraction = memory_fraction(length, page, pages, reuse)
        load = memory_load(length, page, pages, reuse, width)
        self.assertAlmostEqual(load / (2.0 * width * length), fraction)

    def test_optima(self):
        length, pages, reuse = 4096.0, 32.0, 0.4
        stated = paper_page_size(length, pages)
        self.assertAlmostEqual(
            memory_fraction(length, stated, pages, reuse),
            fraction_at_paper_page_size(length, pages, reuse),
        )
        best = minimizing_page_size(length, pages, reuse)
        best_fraction = minimized_memory_fraction(length, pages, reuse)
        self.assertAlmostEqual(memory_fraction(length, best, pages, reuse), best_fraction)
        self.assertGreaterEqual(memory_fraction(length, best * 0.5, pages, reuse) + 1e-12, best_fraction)
        self.assertGreaterEqual(memory_fraction(length, best * 2.0, pages, reuse) + 1e-12, best_fraction)

    def test_gap_bound_is_positive(self):
        self.assertGreater(approximation_gap_bound(64, 0.2, 16), 0.0)

    def test_resolve_k(self):
        self.assertEqual(resolve_k(100, 16, 0.3, None), 30)
        self.assertEqual(resolve_k(2048, 16, 1.0, 2048), 128)
        self.assertEqual(resolve_k(4, 16, 0.3, None), 1)
        self.assertEqual(resolve_k(0, 16, 1.0, None), 0)


class AttentionTest(unittest.TestCase):
    def test_hand_scored_box(self):
        query = torch.tensor([1.0, -2.0, 0.0, 0.5]).view(1, 1, 1, 4)
        lower = torch.tensor([-1.0, -3.0, 0.0, 1.0]).view(1, 1, 1, 4)
        upper = torch.tensor([2.0, 4.0, 1.0, 3.0]).view(1, 1, 1, 4)
        score = bounding_box_scores(query, lower, upper)
        self.assertTrue(torch.allclose(score, torch.tensor(9.5)))

    def test_box_is_an_upper_bound(self):
        torch.manual_seed(0)
        keys = torch.randn(17, 8)
        query = torch.randn(8)
        vertex = torch.where(query >= 0, keys.amax(0), keys.amin(0))
        keys = torch.cat([keys, vertex.unsqueeze(0)], dim=0)
        lower = keys.amin(0).view(1, 1, 1, -1)
        upper = keys.amax(0).view(1, 1, 1, -1)
        score = bounding_box_scores(query.view(1, 1, 1, -1), lower, upper).reshape(())
        dots = keys @ query
        self.assertGreaterEqual(float(score + 1e-4), float(dots.max()))
        self.assertTrue(torch.allclose(score, dots.max()))

    def test_full_budget_matches_dense(self):
        torch.manual_seed(1)
        query = torch.randn(2, 4, 10, 8)
        key = torch.randn(2, 2, 10, 8)
        value = torch.randn(2, 2, 10, 8)
        scale = 1.0 / math.sqrt(8)
        dense = dense_causal_attention(query, key, value, scale)
        sparse, _ = sparse_causal_attention(
            query, key, value, page_size=4, k_pages=3, n_recent=1, scale=scale
        )
        self.assertTrue(torch.allclose(dense, sparse, atol=1e-5, rtol=1e-5))

    def test_recent_page_is_kept_under_a_tight_budget(self):
        torch.manual_seed(2)
        tokens, page = 32, 8
        query = torch.randn(1, 1, 1, 8)
        key = torch.randn(1, 1, tokens, 8)
        value = torch.randn(1, 1, tokens, 8)
        _, selected = sparse_causal_attention(
            query,
            key,
            value,
            page_size=page,
            k_pages=1,
            n_recent=1,
            scale=1.0,
            query_positions=torch.tensor([[tokens - 1]]),
        )
        kept = selected[selected >= 0].unique().tolist()
        self.assertEqual(kept, [(tokens - 1) // page])

    def test_several_pinned_pages_still_keep_the_newest(self):
        torch.manual_seed(2)
        tokens, page = 32, 8
        query = torch.randn(1, 1, 1, 8)
        key = torch.randn(1, 1, tokens, 8)
        value = torch.randn(1, 1, tokens, 8)
        _, selected = sparse_causal_attention(
            query,
            key,
            value,
            page_size=page,
            k_pages=1,
            n_recent=3,
            scale=1.0,
            query_positions=torch.tensor([[tokens - 1]]),
        )
        kept = selected[selected >= 0].unique().tolist()
        self.assertEqual(kept, [(tokens - 1) // page])

    def test_prune_and_approx_stay_finite(self):
        torch.manual_seed(3)
        query = torch.randn(1, 2, 8, 8)
        key = torch.randn(1, 2, 8, 8)
        value = torch.randn(1, 2, 8, 8)
        exact, _ = sparse_causal_attention(
            query, key, value, page_size=4, k_pages=2, n_recent=1, scale=0.25
        )
        pruned, _ = sparse_causal_attention(
            query,
            key,
            value,
            page_size=4,
            k_pages=2,
            n_recent=1,
            scale=0.25,
            prune_threshold=0.5,
        )
        approx, _ = sparse_causal_attention(
            query,
            key,
            value,
            page_size=4,
            k_pages=2,
            n_recent=1,
            scale=0.25,
            approx=True,
        )
        self.assertTrue(torch.isfinite(pruned).all())
        self.assertTrue(torch.isfinite(approx).all())
        self.assertFalse(torch.allclose(exact, pruned, atol=1e-5))
        self.assertFalse(torch.allclose(exact, approx, atol=1e-4))

    def test_unselected_page_gets_no_gradient(self):
        torch.manual_seed(4)
        page = 4
        query = torch.randn(1, 1, 1, 8, requires_grad=True)
        key = torch.randn(1, 1, 16, 8, requires_grad=True)
        value = torch.randn(1, 1, 16, 8, requires_grad=True)
        out, _ = sparse_causal_attention(
            query,
            key,
            value,
            page_size=page,
            k_pages=1,
            n_recent=1,
            scale=1.0,
            query_positions=torch.tensor([[15]]),
        )
        out.sum().backward()
        self.assertEqual(int(torch.count_nonzero(key.grad[:, :, :12])), 0)
        self.assertGreater(float(key.grad[:, :, 12:].abs().sum()), 0.0)
        importance = page_importance(key.grad, page)
        mask = retain_mask(importance, 0.25)
        self.assertTrue(bool(mask[0, 0, -1]))
        self.assertEqual(int(mask.sum()), 1)


class CacheTest(unittest.TestCase):
    def _keys(self, storage: str):
        config = _tiny_config(n_layer=1, n_kv_head=4, n_head=4, kv_storage=storage, block_size=32)
        batch, heads, tokens, dim = 2, config.n_kv_head, 10, config.head_dim
        key = torch.randn(batch, heads, tokens, dim)
        value = torch.randn(batch, heads, tokens, dim)
        return config, key, value

    def test_incremental_bounds_match_a_full_write(self):
        for storage in ("fp32", "fp16"):
            config, key, value = self._keys(storage)
            written = PagedKVCache.create(config, key.shape[0], torch.device("cpu"))
            stepped = PagedKVCache.create(config, key.shape[0], torch.device("cpu"))
            written.write_sequence(0, key, value)
            for token in range(key.shape[2]):
                stepped.append(0, key[:, :, token], value[:, :, token])
                stepped.advance()
            self.assertTrue(torch.equal(written.k, stepped.k))
            self.assertTrue(torch.equal(written.page_lens, stepped.page_lens))
            self.assertTrue(torch.equal(written.num_pages, stepped.num_pages))
            self.assertTrue(torch.allclose(written.kmin, stepped.kmin))
            self.assertTrue(torch.allclose(written.kmax, stepped.kmax))
            lower, upper = page_bounds(written.k[0].float(), written.page_lens[0])
            self.assertTrue(torch.allclose(lower, written.kmin[0]))
            self.assertTrue(torch.allclose(upper, written.kmax[0]))

    def test_int8_roundtrip_matches_bulk_quantization(self):
        config, key, value = self._keys("int8")
        written = PagedKVCache.create(config, key.shape[0], torch.device("cpu"))
        stepped = PagedKVCache.create(config, key.shape[0], torch.device("cpu"))
        written.write_sequence(0, key, value)
        for token in range(key.shape[2]):
            stepped.append(0, key[:, :, token], value[:, :, token])
            stepped.advance()
        self.assertTrue(torch.equal(written.k_i8, stepped.k_i8))
        self.assertTrue(torch.allclose(written.k_scale, stepped.k_scale))
        page = config.page_size
        slots = math.ceil(key.shape[2] / page)
        scale = written.k_scale[0]
        recon = written.k_i8[0].float() * scale[:, :, :, None, None]
        flat = recon[:, :, :slots].reshape(key.shape[0], key.shape[1], slots * page, key.shape[-1])
        flat = flat[:, :, : key.shape[2]]
        token_scale = scale[:, :, :slots].repeat_interleave(page, dim=2)[:, :, : key.shape[2]]
        err = (flat - key).abs()
        self.assertTrue(torch.all(err <= 0.5 * token_scale.unsqueeze(-1) + 1e-4))

    def test_append_past_block_size_raises(self):
        config = _tiny_config(n_layer=1, n_kv_head=4, n_head=4, block_size=4, page_size=4)
        cache = PagedKVCache.create(config, 1, torch.device("cpu"))
        key = torch.randn(1, 4, 4, config.head_dim)
        cache.write_sequence(0, key, key)
        with self.assertRaises(RuntimeError):
            cache.append(0, key[:, :, 0], key[:, :, 0])


class SystemTest(unittest.TestCase):
    def test_prefill_plus_decode_matches_dense_forward(self):
        torch.manual_seed(5)
        config = _tiny_config()
        model = TinyServeLM(config).eval()
        tokens = torch.randint(0, config.vocab_size, (2, 9))
        with torch.no_grad():
            full = model.forward(tokens, dense=True)
            cache = PagedKVCache.create(config, tokens.shape[0], tokens.device)
            model.prefill(tokens[:, :-1], cache)
            step = model.decode_token(tokens[:, -1], cache)
        self.assertTrue(torch.allclose(full[:, -1], step[:, -1], atol=1e-4, rtol=1e-4))

    def test_generate_is_deterministic_and_can_stop_early(self):
        torch.manual_seed(6)
        config = _tiny_config(top_k_ratio=0.5)
        model = TinyServeLM(config).eval()
        prompt = torch.randint(0, config.vocab_size, (2, 6))
        first = generate(model, prompt, 4, temperature=0.0)
        second = generate(model, prompt, 4, temperature=0.0)
        self.assertTrue(torch.equal(first.tokens, second.tokens))
        self.assertTrue(torch.equal(first.lengths, torch.full((2,), 10)))
        self.assertGreaterEqual(first.page_hit_rate, 0.0)
        self.assertLessEqual(first.page_hit_rate, 1.0)
        self.assertIn("prefill", first.profile)
        self.assertIn("decode", first.profile)
        self.assertTrue(model.training is False)

        config.entropy_threshold = 1e9
        stopped = generate(model, prompt, 4, temperature=0.0)
        self.assertTrue(stopped.stopped_early)
        self.assertTrue(torch.equal(stopped.lengths, torch.full((2,), 6)))

    def test_entropy_of_a_peaked_distribution_is_small(self):
        logits = torch.tensor([[0.0, 80.0, 0.0]])
        self.assertLess(float(token_entropy(logits)), 1e-6)
        same = torch.randn(2, 3, 4)
        self.assertEqual(page_overlap(same, same), 1.0)

    def test_training_backward_and_grad_sync_noop(self):
        torch.manual_seed(7)
        config = _tiny_config(checkpoint=True, dense_train=False, top_k_ratio=1.0)
        model = TinyServeLM(config).train()
        tokens = torch.randint(0, config.vocab_size, (2, 8))
        logits = model(tokens)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, config.vocab_size), tokens[:, 1:].reshape(-1))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.wte.weight.grad)
        self.assertGreater(float(model.wte.weight.grad.abs().sum()), 0.0)
        before = model.wte.weight.grad.clone()
        AsyncGradSync().sync(model.parameters())
        self.assertTrue(torch.equal(before, model.wte.weight.grad))

    def test_scheduler_moves_only_when_it_helps(self):
        scheduler = ServingScheduler(2)
        self.assertEqual(scheduler.admit(1, 10), 0)
        self.assertEqual(scheduler.admit(2, 10), 1)
        self.assertEqual(scheduler.admit(3, 50), 0)
        moved = scheduler.migrate(0.25)
        self.assertEqual(moved, [(1, 0, 1)])
        self.assertLess(scheduler.imbalance(), 0.7)

        stuck = ServingScheduler(2)
        stuck.admit(1, 100)
        stuck.admit(2, 1)
        self.assertEqual(stuck.migrate(), [])

    def test_batch_dispatch_splits_and_rejoins(self):
        batch = torch.arange(5)
        out = dispatch_batch(lambda chunk: chunk * 2, batch, [torch.device("cpu"), torch.device("cpu")])
        self.assertTrue(torch.equal(out, batch * 2))

    def test_unknown_preset(self):
        with self.assertRaises(KeyError):
            preset("no-such-model")
        self.assertEqual(preset("gpt2-345m").n_embd, 1024)


if __name__ == "__main__":
    unittest.main()
