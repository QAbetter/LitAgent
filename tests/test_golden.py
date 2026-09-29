"""test_golden.py：golden 快照机制的单测（全程桩，不联网）。

验证三件事：
  1) 录制逻辑（record_for）能正确收集向量/改写/精排分数；
  2) save/load 往返一致、回放桩返回预录值且**不联网**；
  3) 用回放桩驱动 HybridRetriever 能离线跑出**确定性**检索结果。
"""

import pytest

from litagent.eval import golden
from litagent.rag.hybrid import HybridRetriever


@pytest.fixture
def fake_model(monkeypatch):
    """把所有外部调用（embed/rewrite/rerank）替换成确定性桩（8 维对齐 conftest）。"""
    monkeypatch.setattr(golden, "embed_query", lambda q: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        golden, "rewrite_query", lambda q: (f"rew-{q}", "note")
    )
    monkeypatch.setattr(
        golden, "rerank", lambda q, c: [5.0] * len(c)
    )


def test_record_collects_hooks(tmp_store, fake_model):
    queries = [type("Q", (), {"question": "BERT pre-train"})(),
               type("Q", (), {"question": "Transformer attention"})()]
    snap = golden.record_for(tmp_store, queries)
    # 改写映射 raw->rewritten
    assert snap["rewrites"]["BERT pre-train"] == "rew-BERT pre-train"
    # 向量/精排按「改写后 query」记录
    assert "rew-BERT pre-train" in snap["vectors"]
    assert isinstance(snap["reranks"].get("rew-BERT pre-train"), list)


def test_save_load_roundtrip(tmp_path):
    snap = {"rewrites": {"q": "r"}, "vectors": {"r": [1.0]}, "reranks": {"r": [5]}}
    path = tmp_path / "nested" / "g.json"
    golden.save(snap, path)
    assert golden.load(path) == snap


def test_playback_returns_recorded_values(tmp_store, fake_model):
    queries = [type("Q", (), {"question": "BERT pre-train"})()]
    snap = golden.record_for(tmp_store, queries)
    emb, rw, rr = golden.playback(snap)

    # 回放桩返回预录值，且不含任何网络调用（这些桩只是查 dict）
    assert emb("rew-BERT pre-train") == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert rw("BERT pre-train") == ("rew-BERT pre-train", "回放 golden")
    assert isinstance(rr("rew-BERT pre-train", []), list)
    # 改写查询若未录制，则原样返回
    assert rw("gibberish") == ("gibberish", "回放 golden")


def test_offline_search_is_deterministic(tmp_store, fake_model):
    """用 golden 回放桩驱动检索，两次结果应完全一致（确定性、离线）。"""
    queries = [type("Q", (), {"question": "BERT parameters layers"})(),
               type("Q", (), {"question": "Transformer attention"})()]
    snap = golden.record_for(tmp_store, queries)
    emb, rw, rr = golden.playback(snap)

    def run():
        r = HybridRetriever(tmp_store, use_rewrite=True, use_bm25=True, use_rerank=True,
                            embed_fn=emb, rewrite_fn=rw, rerank_fn=rr)
        return [[(x["id"], round(x["score"], 6)) for x in r.search(q.question, top_k=2)]
                for q in queries]

    assert run() == run()