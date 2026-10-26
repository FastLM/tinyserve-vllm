import random
import unittest

from tinyserve.selection import (
    bonus_select,
    same_page_set,
    streaming_select,
    streaming_topk,
)


class StreamingTopKTest(unittest.TestCase):
    def test_matches_stable_argsort(self):
        rng = random.Random(0)
        for _ in range(40):
            n = rng.randint(1, 300)
            k = rng.randint(1, min(256, n))
            scores = [rng.uniform(-5, 5) for _ in range(n)]
            if n > 4:
                scores[3] = scores[1]
            got = [ident for _, ident in streaming_topk(scores, list(range(n)), k)]
            expect = sorted(range(n), key=lambda i: (-scores[i], i))[:k]
            self.assertEqual(got, expect)

    def test_tie_break_prefers_smaller_page(self):
        scores = [1.0, 2.0, 2.0, 0.0, 2.0]
        got = [ident for _, ident in streaming_topk(scores, list(range(5)), 4)]
        self.assertEqual(got, [1, 2, 4, 0])

    def test_short_list_is_padded(self):
        got = [ident for _, ident in streaming_topk([0.2, 1.0, -3.0], list(range(3)), 5)]
        self.assertEqual(got[:3], [1, 0, 2])
        self.assertEqual(got[3:], [-1, -1])

    def test_pin_matches_score_bonus(self):
        rng = random.Random(1)
        for _ in range(30):
            n = rng.randint(1, 80)
            k = rng.randint(1, min(16, n))
            n_recent = rng.randint(0, min(4, n))
            scores = [rng.random() + i * 1e-3 for i in range(n)]
            pinned = streaming_select(scores, k, n_recent, tile=8)
            boosted = bonus_select(scores, k, n_recent)
            self.assertTrue(same_page_set(pinned, boosted), (pinned, boosted, n_recent, k))


if __name__ == "__main__":
    unittest.main()
