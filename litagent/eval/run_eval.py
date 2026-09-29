"""Phase 4（扩展版）：检索评测入口——4 档消融对比。

对比四种检索配置，量化每个组件的增量：
  1. 纯向量          use_rewrite=False, use_bm25=False, use_rerank=False
  2. +BM25           use_rewrite=False, use_bm25=True , use_rerank=False
  3. +跨语言改写      use_rewrite=True , use_bm25=True , use_rerank=False
  4. +LLM 精排       use_rewrite=True , use_bm25=True , use_rerank=True

评测集按 PDF 文件名（stem）自动匹配，解决「换论文却仍用 BERT 评测集」的张冠李戴问题。

用法：
    python -m litagent.eval.run_eval [pdf路径] [top_k]

示例：
    python -m litagent.eval.run_eval                        # 默认 data/bert.pdf, top_k=4
    python -m litagent.eval.run_eval data/paper.pdf         # 评测 Transformer 论文
    python -m litagent.eval.run_eval data/bert.pdf 4
"""

import sys
from pathlib import Path

from litagent.eval.dataset import queries_for
from litagent.eval.metrics import evaluate, summarize
from litagent.rag.hybrid import HybridRetriever
from litagent.rag.pipeline import build_index
from litagent.rag.vector_store import VectorStore

# 四档消融的标签与检索器参数（按「从简到全」排列）
CONFIGS: list[tuple[str, dict]] = [
    ("纯向量", dict(use_rewrite=False, use_bm25=False, use_rerank=False)),
    ("+BM25", dict(use_rewrite=False, use_bm25=True, use_rerank=False)),
    ("+跨语言改写", dict(use_rewrite=True, use_bm25=True, use_rerank=False)),
    ("+LLM精排", dict(use_rewrite=True, use_bm25=True, use_rerank=True)),
]


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


def _mark(score: float) -> str:
    """把 0/1 指标显示成 ✓/✗，命中的小数值显示为两位小数。"""
    return "✓" if score > 0 else "·"


def _fmt_pages(pages: list[int], gt: frozenset[int], k: int) -> str:
    """把命中页列表格式化，命中的 GT 页用括号标出。"""
    shown = []
    for p in pages[:k]:
        shown.append(f"({p})" if p in gt else str(p))
    return ",".join(shown)


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/bert.pdf"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    stem = Path(pdf_path).stem
    queries = queries_for(stem)  # 按论文名自动匹配，张冠李戴会在此抛 KeyError
    store = _load_store(pdf_path)

    print(f"评测对象: {pdf_path}   评测集 {len(queries)} 条（中英对照）  top_k={top_k}")
    cats = {}
    for q in queries:
        cats[q.category] = cats.get(q.category, 0) + 1
    print(f"类型分布: " + "  ".join(f"{c}×{n}" for c, n in sorted(cats.items())))

    # 每个配置跑一遍全部样本，记录每条的指标与命中页
    results: list[tuple[str, list[dict], list[list[int]]]] = []
    for label, cfg in CONFIGS:
        retriever = HybridRetriever(store, **cfg)
        rows, page_lists = [], []
        for q in queries:
            pages = _pages(retriever.search(q.question, top_k=top_k))
            rows.append(evaluate(pages, set(q.gt_pages), top_k))
            page_lists.append(pages)
        results.append((label, rows, page_lists))

    # 逐条明细
    print("\n" + "=" * 112)
    header = f"{'问题':<30}{'GT':<9}{'类型':<6}" + "".join(
        f"{lbl:<16}" for lbl, _, _ in results
    )
    print(header)
    print("-" * 112)
    for i, q in enumerate(queries):
        line = f"{q.question[:28]:<30}{','.join(map(str, sorted(q.gt_pages))):<9}{q.category:<6}"
        for _, rows, page_lists in results:
            hit = _mark(rows[i]["hit"])
            pages = _fmt_pages(page_lists[i], q.gt_pages, top_k)
            line += f"{hit}{pages:<15}"
        print(line)

    # 汇总指标对比
    print("\n" + "=" * 112)
    sums = [summarize(rows) for _, rows, _ in results]
    base = sums[0]
    print(f"{'指标':<12}" + "".join(f"{lbl:<14}" for lbl, _, _ in results) + f"{'相对基线':<12}")
    print("-" * 112)
    for key, label in (("hit", "Hit Rate"), ("recall", "Recall@k"), ("mrr", "MRR")):
        line = f"{label:<12}"
        for s in sums:
            line += f"{s[key]:<14.3f}"
        line += f"{sums[-1][key] - base[key]:+.3f}"
        print(line)

    # 按类型分层看表现（只看最终「+LLM精排」配置 vs 基线「纯向量」）
    print("\n按问题类型分层（左=纯向量基线，右=+LLM精排）：")
    print("-" * 112)
    by_cat: dict[str, tuple[list[dict], list[dict]]] = {}
    for i, q in enumerate(queries):
        base_rows, full_rows = by_cat.setdefault(q.category, ([], []))
        base_rows.append(results[0][1][i])
        full_rows.append(results[-1][1][i])
    for cat in sorted(by_cat):
        b, f = by_cat[cat]
        bs, fs = summarize(b), summarize(f)
        print(
            f"{cat:<6}  命中率 {bs['hit']:.2f}→{fs['hit']:.2f}   "
            f"Recall@k {bs['recall']:.2f}→{fs['recall']:.2f}   "
            f"MRR {bs['mrr']:.2f}→{fs['mrr']:.2f}"
        )


if __name__ == "__main__":
    main()