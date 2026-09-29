"""Phase 1：检索器（把问题向量化后召回相关文本块）。"""

from litagent.rag.embedding import embed_query
from litagent.rag.vector_store import VectorStore


def retrieve(question: str, store: VectorStore, top_k: int = 4) -> list[dict]:
    """给定问题，从向量库召回最相关的文本块。

    Args:
        question: 用户问题。
        store: 已建好索引的向量库。
        top_k: 召回数量。

    Returns:
        与 VectorStore.search 相同结构的列表，每项含 text/page_no/score。
    """
    query_embedding = embed_query(question)
    return store.search(query_embedding, top_k=top_k)