"""Phase 4：可配置的检索管线——跨语言改写 + BM25/向量混合召回 + LLM 精排。

设计成可组合的管线，便于消融实验对比每个组件的增量：
- use_rewrite：跨语言查询改写（中文问题 → 英文检索词）；
- use_bm25：BM25 词面检索与向量检索做 RRF 融合；
- use_rerank：召回后用文本模型对候选做精排。

默认配置（use_rewrite=True, use_bm25=True, use_rerank=False）即生产检索路径，
也是 agent.tools 中 search_text 所用。

依赖注入：`embed_fn / rewrite_fn / rerank_fn` 三个函数可替换，默认真实实现。
评测（golden 回放）与单元测试注入确定性桩，让整条检索管线离线、可复现。
"""

import json
import re
from collections.abc import Callable

from rank_bm25 import BM25Okapi

from config.settings import settings
from litagent.llm import get_client
from litagent.rag.embedding import embed_query
from litagent.rag.vector_store import VectorStore

# 英文词面分词：小写 + 提取字母数字串，与 BM25 词面匹配目标一致
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# 检测是否含中日韩文字（用于判断是否需要跨语言改写）
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

# RRF 融合常数，越大越偏向「排名靠前」的记录
RRF_K = 60

# 可注入的函数签名
EmbedFn = Callable[[str], list[float]]
RewriteFn = Callable[[str], tuple[str, str]]
RerankFn = Callable[[str, list[dict]], list[float] | None]


def _tokenize(text: str) -> list[str]:
    """把整段英文文本切成词面 token（BM25 用）。"""
    return _TOKEN_RE.findall(text.lower())


def rewrite_query(question: str) -> tuple[str, str]:
    """跨语言查询改写：把中文问题翻译成适合检索英文论文的英文 query。

    Returns:
        (改写后的 query, 说明)。若不含中文，返回 (原问题, "无需改写")。
    """
    if not _CJK_RE.search(question):
        return question, "无需改写"
    resp = get_client().chat.completions.create(
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


def rerank(query: str, candidates: list[dict]) -> list[float] | None:
    """用文本模型给候选片段打分（0-5），返回与 candidates 对齐的分数列表。

    解析失败（模型返回非 JSON）返回 None，调用方回退到原召回顺序。
    """
    if not candidates:
        return []
    # 候选片段本身较短（≤ chunk_size），喂完整文本，避免截断丢掉关键句
    numbered = "\n".join(f"[{i}] {c['text']}" for i, c in enumerate(candidates))
    resp = get_client().chat.completions.create(
        model=settings.text_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是检索结果精排器。评估每个候选片段与问题的相关性，"
                    "给出 0-5 的整数分（5=高度相关，0=无关）。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{query}\n\n候选片段：\n{numbered}\n\n"
                    '请只输出一个 JSON 对象，键为片段编号（字符串），值为 0-5 整数分。'
                    '例如：{"0": 5, "1": 2}'
                ),
            },
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    # 剥离可能的 markdown 代码块包裹
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    scores: list[float] = []
    for i in range(len(candidates)):
        try:
            scores.append(float(data.get(str(i), 0)))
        except (TypeError, ValueError):
            scores.append(0.0)
    return scores


class HybridRetriever:
    """可配置检索器：改写 → 召回（向量 / 向量+BM25）→ 精排（可选 LLM rerank）。

    三个外部调用点（向量化 / 改写 / 精排）默认使用真实实现，可通过构造参数
    注入确定性桩，从而在无需联网的情况下复现同一检索路径。
    """

    def __init__(
        self,
        store: VectorStore,
        candidate_k: int = 20,
        use_rewrite: bool = True,
        use_bm25: bool = True,
        use_rerank: bool = False,
        embed_fn: EmbedFn = embed_query,
        rewrite_fn: RewriteFn = rewrite_query,
        rerank_fn: RerankFn = rerank,
    ):
        self.store = store
        self.candidate_k = candidate_k
        self.use_rewrite = use_rewrite
        self.use_bm25 = use_bm25
        self.use_rerank = use_rerank
        self.embed_fn = embed_fn
        self.rewrite_fn = rewrite_fn
        self.rerank_fn = rerank_fn
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []
        # 记录最近一次检索的原始问题与改写结果，供评测/调试查看
        self.last_query = ""
        self.last_rewritten = ""

    def _ensure_bm25(self) -> None:
        """首次混合检索时从向量库取出全部块，离线构建 BM25 词面索引。"""
        if self._bm25 is not None:
            return
        self._chunks = self.store.get_all()
        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in self._chunks])

    def _recall(self, query: str, k: int) -> list[dict]:
        """召回阶段：纯向量，或 BM25 + 向量 RRF 融合。"""
        if not self.use_bm25:
            return self.store.search(self.embed_fn(query), top_k=k)

        self._ensure_bm25()
        vec_hits = self.store.search(self.embed_fn(query), top_k=k)
        bm25_scores = self._bm25.get_scores(_tokenize(query))

        rrf: dict[str, dict] = {}
        for rank, hit in enumerate(vec_hits, start=1):
            item = dict(hit)
            item["rrf"] = 1.0 / (RRF_K + rank)
            rrf[hit["id"]] = item
        ranked_bm25 = sorted(
            range(len(self._chunks)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:k]
        for rank, idx in enumerate(ranked_bm25, start=1):
            c = self._chunks[idx]
            item = rrf.setdefault(
                c["id"],
                {
                    "id": c["id"],
                    "text": c["text"],
                    "page_no": c["page_no"],
                    "chunk_index": c["chunk_index"],
                    "rrf": 0.0,
                },
            )
            item["rrf"] += 1.0 / (RRF_K + rank)

        ranked = sorted(rrf.values(), key=lambda x: x["rrf"], reverse=True)[:k]
        for x in ranked:
            x["score"] = x.pop("rrf")
        return ranked

    def _llm_rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """精排阶段：用 LLM 打分重排，失败则保持原顺序。"""
        scores = self.rerank_fn(query, candidates)
        if scores is None:
            return candidates[:top_k]
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        out: list[dict] = []
        for i in order[:top_k]:
            c = dict(candidates[i])
            c["score"] = scores[i]
            out.append(c)
        return out

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """执行检索管线，返回与 VectorStore.search 相同结构的结果列表。"""
        q, _ = (self.rewrite_fn(query) if self.use_rewrite else (query, "无需改写"))
        self.last_query = query
        self.last_rewritten = q

        recall_k = self.candidate_k if self.use_rerank else top_k
        results = self._recall(q, recall_k)

        if self.use_rerank and results:
            results = self._llm_rerank(q, results, top_k)

        return results[:top_k]