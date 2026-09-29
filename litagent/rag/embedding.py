"""Phase 1：文本向量化（调用智谱 embedding 模型）。

走 OpenAI 兼容接口的 /embeddings 端点，模型名默认 embedding-3（2048 维）。
"""

from config.settings import settings
from litagent.llm import get_client

# 智谱 embedding 接口单次最多接受 64 条文本，分批处理
_BATCH_SIZE = 64


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把一批文本转成向量（自动分批）。

    Args:
        texts: 待向量化的文本列表。

    Returns:
        与输入顺序一一对应的向量列表。
    """
    if not texts:
        return []
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        resp = get_client().embeddings.create(model=settings.embedding_model, input=batch)
        embeddings.extend(item.embedding for item in resp.data)
    return embeddings


def embed_query(text: str) -> list[float]:
    """把单个查询文本转成向量。"""
    return embed_texts([text])[0]