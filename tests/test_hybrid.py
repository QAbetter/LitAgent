"""test_hybrid.py：混合检索管线的集成测试（全部注入确定性桩，不联网）。

用 conftest.tmp_store 建一个 4 页的小向量库，验证三条关键路径：
  1) 纯向量检索；
  2) BM25 + 向量 RRF 融合（词面信号能兜底语义信号）；
  3) 跨语言改写被调用、LLM 精排被打分重排。
同时作为 r1 依赖注入（embed_fn/rewrite_fn/rerank_fn 可替换）的功能验证。
"""

import pytest

from litagent.rag.hybrid import HybridRetriever

# 一段与某块高度词面重叠但语义桩未必命中的查询，用于验证 BM25 兜底
BM25_QUERY = "BERT parameters layers"


def _embed_stub(text: str) -> list[float]:
    """把文本编码成确定性向量：前若干字符的序号归一化到单元 8 维。"""
    vec = [0.0] * 8
    for i, ch in enumerate(text):
        vec[i % 8] += ord(ch)
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _make(store, **overrides):
    params = dict(
        use_rewrite=False,
        use_bm25=True,
        use_rerank=False,
        embed_fn=_embed_stub,
        rewrite_fn=lambda q: (q, "noop"),
        rerank_fn=lambda q, c: None,
    )
    params.update(overrides)
    return HybridRetriever(store, **params)


def test_hybrid_returns_sorted_results(tmp_store):
    r = _make(tmp_store)
    results = r.search("pre-train BooksCorpus Wikipedia", top_k=3)
    assert results
    assert 1 <= len(results) <= 3
    # 分数单调不增（已排序）
    scores = [x["score"] for x in results]
    assert all(a >= b for a, b in zip(scores, scores[1:], strict=False))
    # BM25 词面能命中含 "parameters/layers" 的块
    hit_texts = " ".join(x["text"] for x in r.search(BM25_QUERY, top_k=2))
    assert "parameters" in hit_texts


def test_rewrite_fn_is_invoked(tmp_store):
    seen = []
    r = _make(
        tmp_store,
        use_rewrite=True,
        rewrite_fn=lambda q: (seen.append(q) or "rewritten-query", "recorded"),
    )
    r.search("中文问题")
    assert seen == ["中文问题"]


def test_rerank_reorders_by_scores(tmp_store):
    # 强制精排把第一个候选排到最后，验证重排生效
    def rerank_fn(query, candidates):
        return [0.0] * len(candidates) if not candidates else [0.0] * (len(candidates) - 1) + [5.0]

    r = _make(tmp_store, use_rerank=True, rerank_fn=rerank_fn, candidate_k=4)
    results = r.search("BERT", top_k=1)
    assert results, "应至少返回一个结果"
    # 精排后唯一返回的应该是原本得分最差的
    raw = _make(tmp_store).search("BERT", top_k=4)
    assert results[0]["id"] == raw[-1]["id"]


def test_rerank_failure_falls_back_to_recall_order(tmp_store):
    # rerank 返回 None（解析失败）时保持原召回顺序
    r = _make(tmp_store, use_rerank=True, rerank_fn=lambda q, c: None, candidate_k=4)
    ordered = r.search("BERT", top_k=4)
    raw = _make(tmp_store).search("BERT", top_k=4)
    assert [x["id"] for x in ordered] == [x["id"] for x in raw]


def test_pure_vector_path(tmp_store):
    r = _make(tmp_store, use_bm25=False)
    results = r.search("Transformer attention parallelizes", top_k=2)
    assert results
    # 唯一含 attention 的块应进入纯向量 top-2（语义相近）
    assert any("attention" in x["text"] for x in results)


def test_invalid_top_k_raised(tmp_store):
    # Chroma 不允许 n_results<=0，检索器把非法 top_k 暴露为异常而非静默
    r = _make(tmp_store)
    with pytest.raises((TypeError, ValueError)):
        r.search("anything", top_k=0)