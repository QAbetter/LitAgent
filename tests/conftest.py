"""pytest 共享夹具：临时向量库 + 确定性向量桩。

所有测试**不联网**：向量、LLM 调用都用确定性桩替换，保证 CI 上可跑、可复现。
"""

import pytest

from litagent.rag.vector_store import VectorStore


def deterministic_embed(text: str, dim: int = 8) -> list[float]:
    """把文本哈希到 dim 维向量：确定性、词面相关（同词元相近）。"""
    vec = [0.0] * dim
    for i, ch in enumerate(text):
        vec[i % dim] += ord(ch)
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def build_store(tmp_path, pages: list[tuple[str, int]]) -> VectorStore:
    """按 (text, page_no) 建一个临时 Chroma 向量库，向量用 deterministic_embed。"""
    store = VectorStore(str(tmp_path / "db"))
    store.collection.add(
        ids=[f"p{p}-c{i}" for i, (_, p) in enumerate(pages)],
        documents=[t for t, _ in pages],
        embeddings=[deterministic_embed(t) for t, _ in pages],
        metadatas=[{"page_no": p, "chunk_index": i} for i, (_, p) in enumerate(pages)],
    )
    return store


@pytest.fixture
def tmp_store(tmp_path):
    """一个含 4 页文本起止的小型向量库，供检索管线测试。"""
    pages = [
        ("BERT uses Masked Language Model and Next Sentence Prediction", 4),
        ("BERT_base has 12 layers, 768 hidden and 110M parameters", 3),
        ("We pre-train on BooksCorpus (800M words) and English Wikipedia", 5),
        ("the Transformer relies entirely on attention and parallelizes", 3),
    ]
    return build_store(tmp_path, pages)