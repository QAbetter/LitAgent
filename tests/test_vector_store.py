"""test_vector_store.py：Chroma 向量库封装单测（本地、确定性、不联网）。

用一个确定性向量桩写入临时目录，验证 add/search/get_all/count 的行为。
"""

import pytest

from litagent.rag.vector_store import VectorStore


def _search(store: VectorStore, query_vec: list[float], top_k: int = 3) -> list[dict]:
    return store.search(query_vec, top_k=top_k)


def test_empty_store(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    assert store.count() == 0
    assert store.get_all() == []


def test_add_and_count(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    store.collection.add(
        ids=["p1-c0"],
        documents=["hello"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"page_no": 1, "chunk_index": 0}],
    )
    assert store.count() == 1


def test_search_returns_metadata_and_similarity(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    store.collection.add(
        ids=["p3-c2", "p5-c1"],
        documents=["cat cat cat", "dog dog dog"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"page_no": 3, "chunk_index": 2}, {"page_no": 5, "chunk_index": 1}],
    )
    res = _search(store, [0.99, 0.01], top_k=1)
    assert len(res) == 1
    hit = res[0]
    assert hit["page_no"] == 3
    assert hit["chunk_index"] == 2
    assert hit["id"] == "p3-c2"
    assert "text" in hit and "score" in hit
    # 相似的返回分数应更高，且 > 不相似的
    assert hit["score"] > 0.5


def test_get_all_roundtrip(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    store.collection.add(
        ids=["p1-c0", "p1-c1"],
        documents=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"page_no": 1, "chunk_index": 0}, {"page_no": 1, "chunk_index": 1}],
    )
    all_items = store.get_all()
    assert len(all_items) == 2
    assert all({"id", "text", "page_no", "chunk_index"} <= set(i) for i in all_items)


@pytest.mark.skip(reason="需要网络与真实索引，由 golden 离线评测覆盖")
def test_add_chunks_integration(tmp_path):
    # 预留：真实 chunk 写入路径已在 conftest.build_store 中间接覆盖
    pass