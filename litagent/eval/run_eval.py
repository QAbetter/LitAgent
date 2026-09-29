"""Phase 4：检索评测入口——对比「纯向量检索」与「混合检索（改写+BM25+向量）」。

用法：
    python -m litagent.eval.run_eval [pdf路径] [top_k]

示例：
    python -m litagent.eval.run_eval                    # 默认 data/bert.pdf, top_k=4
    python -m litagent.eval.run_eval data/paper.pdf 5
"""

import sys
from pathlib import Path

from litagent.eval.dataset import BERT_QUERIES
from litagent.eval.metrics import evaluate, summarize
from litagent.rag.hybrid import HybridRetriever
from litagent.rag.pipeline import build_index
from litagent.rag.retriever import retrieve
from litagent.rag.vector_store import VectorStore


def _load_store(pdf_path: str) -> VectorStore:
    """复用已有索引，没有就现场建立。"""
    stem = Path(pdf_path).stem
    index_dir = Path("data") / "indexes" / stem
    if index_dir.exists() and any(index_dir.iterdir()):
        return VectorStore(str(index_dir))
    return build_index(pdf_path, str(index_dir))


def _pages(results: list[dict]) -> list[int]:
    """从检索结果里抽出页码列表（按排名顺序）。"""
    return [r["page_no"] for r in results]


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/bert.pdf"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    store = _load_store(pdf_path)
    hybrid = HybridRetriever(store)
    queries = BERT_QUERIES

    print(f"评测对象: {pdf_path}  评测集 {len(queries)} 条  top_k={top_k}\n")

    base_rows: list[dict] = []
    hyb_rows: list[dict] = []
    detail: list[tuple] = []

    for q in queries:
        base_pages = _pages(retrieve(q.question, store, top_k=top_k))
        hyb_pages = _pages(hybrid.search(q.question, top_k=top_k))
        # hybrid.search 内部已完成跨语言改写，改写结果存于 last_rewritten
        base_rows.append(evaluate(base_pages, set(q.gt_pages), top_k))
        hyb_rows.append(evaluate(hyb_pages, set(q.gt_pages), top_k))
        detail.append((q, hybrid.last_rewritten, base_pages, hyb_pages))

    base_sum = summarize(base_rows)
    hyb_sum = summarize(hyb_rows)

    # 逐条对比
    print(f"{'问题':<32}{'GT页':<8}{'改写':<30}{'纯向量命中页':<13}{'混合命中页':<13}")
    print("-" * 100)
    for q, rewritten, base_pages, hyb_pages in detail:
        print(
            f"{q.question[:30]:<32}"
            f"{','.join(map(str, sorted(q.gt_pages))):<8}"
            f"{rewritten[:28]:<30}"
            f"{str(base_pages):<13}"
            f"{str(hyb_pages):<13}"
        )

    # 汇总指标
    print("\n" + "=" * 100)
    print(f"{'指标':<12}{'纯向量检索':<14}{'混合检索(改写+BM25+向量)':<22}{'提升':<8}")
    print("-" * 100)
    labels = {"hit": "Hit Rate", "recall": "Recall@k", "mrr": "MRR"}
    for key, label in labels.items():
        print(
            f"{label:<12}{base_sum[key]:<14.3f}{hyb_sum[key]:<22.3f}"
            f"{hyb_sum[key] - base_sum[key]:+.3f}"
        )


if __name__ == "__main__":
    main()