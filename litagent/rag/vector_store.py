"""Phase 1：向量库（用 ChromaDB 存储和检索向量）。

这里我们自己计算好向量再写入，并显式告诉 ChromaDB 不用内置 embedding 模型
（embedding_function=None），避免它去加载本地模型。
"""

import chromadb

from litagent.rag.chunking import Chunk


class VectorStore:
    """封装 ChromaDB：写入文本块、按向量检索。"""

    def __init__(self, persist_dir: str, collection_name: str = "litagent"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        # 使用余弦相似度作为距离度量（embedding 检索的常用选择）
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """把文本块及其向量写入向量库。"""
        self.collection.add(
            ids=[f"p{c.page_no}-c{c.chunk_index}" for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {"page_no": c.page_no, "chunk_index": c.chunk_index} for c in chunks
            ],
        )

    def search(
        self, query_embedding: list[float], top_k: int = 4
    ) -> list[dict]:
        """按向量检索最相关的 top_k 个块。

        Returns:
            列表，每项为 {"id", "text", "page_no", "chunk_index", "score"}。
            score 是归一化后的相似度，越大越相关。
        """
        res = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        results: list[dict] = []
        for doc, meta, dist in zip(docs, metas, dists, strict=False):
            # 余弦距离 → 余弦相似度
            results.append(
                {
                    "id": f"p{meta['page_no']}-c{meta['chunk_index']}",
                    "text": doc,
                    "page_no": meta["page_no"],
                    "chunk_index": meta["chunk_index"],
                    "score": 1.0 - float(dist),
                }
            )
        return results

    def get_all(self) -> list[dict]:
        """返回库里所有文本块，供混合检索离线构建 BM25 词面索引。

        Returns:
            每项 {"id", "text", "page_no", "chunk_index"}。
        """
        res = self.collection.get()
        ids = res["ids"]
        if not ids:
            return []
        return [
            {
                "id": i,
                "text": d,
                "page_no": m["page_no"],
                "chunk_index": m["chunk_index"],
            }
            for i, d, m in zip(ids, res["documents"], res["metadatas"], strict=False)
        ]

    def count(self) -> int:
        """返回向量库中已写入的条目数。"""
        return self.collection.count()