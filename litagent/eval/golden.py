"""Phase 4（可选）：golden 快照——让评测离线、确定性、可复现。

背景：HybridRetriever.search 默认调用真实 API（embedding + LLM 改写/精排），
导致评测每次联网、非确定性、且烧钱。golden 机制把这些外部调用点「录制」
到本地 JSON，回放时通过依赖注入换成确定性桩，让整条检索管线离线重跑出
与在线一致的结果。CI / 简历演示可以完全离线复现指标。

用法：
    # 联网录制一篇论文的 golden 快照（用到真实 API）
    python -m litagent.eval.golden --record data/bert.pdf

    # 加载桩，构造可离线检索的 HybridRetriever（供 run_eval --offline 使用）
    hooks = playback(load(snapshot_path("bert")))
    retriever = HybridRetriever(store, **cfg, embed_fn=hooks[0], rewrite_fn=hooks[1], rerank_fn=hooks[2])
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from litagent.eval.dataset import queries_for
from litagent.rag.embedding import embed_query
from litagent.rag.hybrid import HybridRetriever, rerank, rewrite_query
from litagent.rag.pipeline import build_index
from litagent.rag.vector_store import VectorStore

# golden 快照目录：data/golden/<stem>.json
GOLDEN_DIR = Path("data") / "golden"


def snapshot_path(paper: str) -> Path:
    """返回某篇论文 golden 快照的文件路径。"""
    return GOLDEN_DIR / f"{paper}.json"


def _load_store(pdf_path: str) -> VectorStore:
    """复用已有索引，没有就现场建立。"""
    stem = Path(pdf_path).stem
    index_dir = Path("data") / "indexes" / stem
    if index_dir.exists() and any(index_dir.iterdir()):
        return VectorStore(str(index_dir))
    return build_index(pdf_path, str(index_dir))


def record_for(store: VectorStore, queries) -> dict:
    """联网跑一遍真实检索，录制三个外部调用点，返回可序列化的快照 dict。

    采用「全开」检索器（use_rewrite + use_bm25 + use_rerank = True），
    并用三个捕获函数包住 embed / rewrite / rerank。

    关键：三个钩子收到的 query 字符串并不相同——rewrite_fn 收到「原始问题」，
    而 embed_fn / rerank_fn 收到「改写后的 query」。因此快照分为三个独立映射：
      - rewrites:  raw -> 改写结果
      - vectors:   改写 query -> 向量
      - reranks:   改写 query -> 精排分数
    回放时按各钩子实际收到的字符串取数，保证与在线路径一致。
    """
    rewrites: dict[str, str] = {}
    vectors: dict[str, list[float]] = {}
    reranks: dict[str, list[float]] = {}

    def cap_embed(q: str) -> list[float]:
        v = embed_query(q)
        vectors[q] = v
        return v

    def cap_rewrite(q: str) -> tuple[str, str]:
        r, note = rewrite_query(q)
        rewrites[q] = r
        return r, note

    def cap_rerank(q: str, candidates: list[dict]):
        scores = rerank(q, candidates)
        if scores is not None:
            reranks[q] = scores
        return scores

    retriever = HybridRetriever(
        store,
        use_rewrite=True,
        use_bm25=True,
        use_rerank=True,
        embed_fn=cap_embed,
        rewrite_fn=cap_rewrite,
        rerank_fn=cap_rerank,
    )
    for q in queries:
        # candidate_k 决定送入精排的候选数，这里跑满
        retriever.search(q.question, top_k=4)

    return {
        "rewrites": rewrites,
        "vectors": vectors,
        "reranks": reranks,
    }


def save(snapshot: dict, path: Path) -> None:
    """把快照写入 JSON（自动建目录）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path) -> dict:
    """从 JSON 读取快照。"""
    return json.loads(path.read_text(encoding="utf-8"))


def playback(snapshot: dict) -> tuple:
    """把快照转成确定性桩，供注入 HybridRetriever。

    Returns:
        (embed_fn, rewrite_fn, rerank_fn)，三者均不联网。
    """
    rewrites = snapshot.get("rewrites", {})
    vectors = snapshot.get("vectors", {})
    reranks = snapshot.get("reranks", {})

    def embed_fn(query: str) -> list[float]:
        vec = vectors.get(query)
        if vec is None:
            # 录制正常时不会发生：query 应已被 cap_embed 以相同字符串记录
            raise ValueError(f"golden 缺少 query 的向量：{query!r}，请重新 --record")
        return vec

    def rewrite_fn(query: str) -> tuple[str, str]:
        return (rewrites.get(query, query), "回放 golden")

    def rerank_fn(query: str, candidates: list[dict]) -> list[float] | None:
        return reranks.get(query)

    return embed_fn, rewrite_fn, rerank_fn


def main() -> None:
    """CLI：--record <pdf> 联网录制 golden；--offline <pdf> 回放验证离线可跑。"""
    args = sys.argv[1:]
    if not args:
        print("用法: python -m litagent.eval.golden --record <pdf> | --offline <pdf>")
        sys.exit(1)

    mode = args[0]
    pdf_path = args[1] if len(args) > 1 else "data/bert.pdf"
    stem = Path(pdf_path).stem
    queries = queries_for(stem)

    if mode == "--record":
        store = _load_store(pdf_path)
        snapshot = record_for(store, queries)
        path = snapshot_path(stem)
        save(snapshot, path)
        print(f"已录制 golden 快照 → {path}  （{len(queries)} 条 query）")
    elif mode == "--offline":
        from litagent.rag.hybrid import HybridRetriever

        store = _load_store(pdf_path)
        snap = load(snapshot_path(stem))
        emb, rw, rr = playback(snap)
        retriever = HybridRetriever(store, use_rewrite=True, use_bm25=True,
                                    use_rerank=True, embed_fn=emb, rewrite_fn=rw, rerank_fn=rr)
        ok = 0
        for q in queries:
            res = retriever.search(q.question, top_k=4)
            if res:
                ok += 1
        print(f"离线回放完成：{ok}/{len(queries)} 条 query 得到结果（未联网）")
    else:
        print(f"未知模式: {mode}")


if __name__ == "__main__":
    main()