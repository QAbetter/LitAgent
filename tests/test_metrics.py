"""test_metrics.py：评测指标的单测（纯逻辑，不联网、确定性）。

覆盖 Hit@k / Recall@k / MRR / evaluate / summarize 的常规与边界情况。
"""

import pytest

from litagent.eval.metrics import (
    evaluate,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize,
)


def test_hit_at_k_hits_in_top():
    assert hit_at_k([3, 2, 1], {2}, 4) == 1.0


def test_hit_at_k_miss():
    assert hit_at_k([3, 2, 1], {9}, 4) == 0.0


def test_hit_at_k_respects_k():
    # 命中在 top_k 之外，不算命中
    assert hit_at_k([5, 4, 3], {3}, 2) == 0.0


def test_recall_at_k_partial():
    # 2 个预期页，只命中 1 个
    assert recall_at_k([3, 4, 5], {3, 6}, 4) == 0.5


def test_recall_at_k_empty_gt():
    assert recall_at_k([1, 2], set(), 4) == 0.0


def test_reciprocal_rank_first_is_one():
    assert reciprocal_rank([2, 3, 4], {2}, 4) == pytest.approx(1.0)


def test_reciprocal_rank_second_is_half():
    assert reciprocal_rank([1, 2, 3], {2}, 4) == pytest.approx(0.5)


def test_reciprocal_rank_miss_is_zero():
    assert reciprocal_rank([1, 2], {9}, 4) == 0.0


def test_reciprocal_rank_beyond_k():
    assert reciprocal_rank([1, 2, 3], {3}, 2) == 0.0


def test_evaluate_shape():
    res = evaluate([2, 5], {2}, 4)
    assert set(res) == {"hit", "recall", "mrr"}


def test_summarize_average():
    rows = [evaluate([2], {2}, 4), evaluate([3], {9}, 4)]
    s = summarize(rows)
    assert s["hit"] == pytest.approx(0.5)
    assert s["mrr"] == pytest.approx(0.5)


def test_summarize_empty():
    assert summarize([]) == {"hit": 0.0, "recall": 0.0, "mrr": 0.0}