"""Phase 4：混合检索器——BM25 词面检索 + 向量语义检索，RRF 融合，并支持跨语言查询改写。

解决两个已实证的检索 bug：
- BUG-1 中英跨语言语义鸿沟：中文 query 直接向量化会因语言之间距离大而检索失败，
  这里先用文本模型把中文问题改写成英文检索 query，再检索英文论文正文。
- BUG-2 检索答偏：纯语义检索遇术语歧义（如「核心贡献」）容易命中无关的「作者分工」块，
  这里叠加 BM25 词面匹配，用 RRF 融合两路排序，取长补短。
"""

import re

from openai import OpenAI
from rank_bm25 import BM25Okapi

from config.settings import settings
from litagent.rag.embedding import embed_query
from litagent.rag.vector_store import VectorStore

# 英文词面分词：小写 + 提取字母数字串，与 BM25 词面匹配目标一致
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# 检测是否含中日韩文字（用于判断是否需要跨语言改写）
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _tokenize(text: str) -> list[str]:
    """把整段英文文本切成词面 token（BM25 用）。"""
    return _TOKEN_RE.findall(text.lower())


def _client() -> OpenAI:
    """创建智谱 OpenAI 兼容客户端。"""
    return OpenAI(api_key=settings.zhipu_api_key, base_url=settings.zhipu_base_url)


def rewrite_query(question: str) -> tuple[str, str]:
    """跨语言查询改写：把中文问题翻译成适合检索英文论文的英文 query。

    Args:
        question: 用户原始问题。

    Returns:
        (改写后的 query, 说明)。若不含中文，返回 (原问题, "无需改写")。
    """
    if not _CJK_RE.search(question):
        return question, "无需改写"
    resp = _client().chat.completions.create(
        model=settings.text_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是论文检索助手的查询改写器。把用户的中文问题翻译成最适合"
                    "检索英文论文正文的英文查询短语（名词/术语优先）。"
                    "只输出英文查询本身，不要解释、不要加引号。"
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    rewritten = (resp.choices[0].message.content or "").strip()
    return (rewritten or question), "中文问题已改写为英文检索词"


class HybridRetriever:
    """BM25 + 向量混合检索器，用 RRF 融合两路排序。BM25 索引首次检索时懒加载。"""

    # RRF 融合常数，越大越偏向「排名靠前」的记录
    RRF_K = 60

    def __init__(self, store: VectorStore, candidate_k: int = 20):
        self.store = store
        self.candidate_k = candidate_k
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []
        # 记录最近一次检索的原始问题与改写结果，供评测/调试查看
        self.last_query = ""
        self.last_rewritten = ""

    def _ensure_bm25(self) -> None:
        """首次检索时从向量库取出全部块，离线构建 BM25 词面索引。"""
        if self._bm25 is not None:
            return
        self._chunks = self.store.get_all()
        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in self._chunks])

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """混合检索：跨语言改写 → 向量 + BM25 两路召回 → RRF 融合 → 取 top_k。

        Returns:
            与 VectorStore.search 相同结构，每项含 id/text/page_no/chunk_index/score
            （score 为 RRF 分数归一化到 [0,1]，越大越相关）。
        """
        rewritten, _ = rewrite_query(query)
        self.last_query = query
        self.last_rewritten = rewritten
        self._ensure_bm25()

        # 两路各自召回 candidate_k 个候选
        vec_hits = self.store.search(embed_query(rewritten), top_k=self.candidate_k)
        bm25_scores = self._bm25.get_scores(_tokenize(rewritten))

        fused: dict[str, dict] = {}

        def add(id_: str, text: str, page_no: int, chunk_index: int, rank: int) -> None:
            item = fused.setdefault(
                id_,
                {"id": id_, "text": text, "page_no": page_no, "chunk_index": chunk_index, "rrf": 0.0},
            )
            item["rrf"] += 1.0 / (self.RRF_K + rank)

        # 向量路：search 已按 score 降序返回，遍历顺序即排名
        for rank, hit in enumerate(vec_hits, start=1):
            add(hit["id"], hit["text"], hit["page_no"], hit["chunk_index"], rank)

        # BM25 路：对全部块按词面得分降序排序，取前 candidate_k 参与融合
        ranked_bm25 = sorted(
            range(len(self._chunks)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[: self.candidate_k]
        for rank, idx in enumerate(ranked_bm25, start=1):
            c = self._chunks[idx]
            add(c["id"], c["text"], c["page_no"], c["chunk_index"], rank)

        # 按融合分降序取 top_k，并把 RRF 分数 min-max 归一化到 [0,1] 作为 score
        ranked = sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)[:top_k]
        raw = [x["rrf"] for x in ranked]
        lo, hi = min(raw), max(raw)
        for x in ranked:
            x["score"] = 1.0 if hi == lo else (x["rrf"] - lo) / (hi - lo)
            del x["rrf"]
        return ranked