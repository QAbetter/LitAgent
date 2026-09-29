"""Phase 1 演示入口：对一篇论文 PDF 全文做问答。

用法：
    python -m litagent.rag.demo <pdf路径> <问题...>

示例：
    python -m litagent.rag.demo data/paper.pdf 这篇论文的核心贡献是什么

首次运行会建立索引（较慢），之后复用同一篇的索引（较快）。
"""

import sys
from pathlib import Path

from litagent.rag.pipeline import ask, build_index
from litagent.rag.vector_store import VectorStore


def _persist_dir(pdf_path: str) -> str:
    """根据 PDF 文件名生成一个向量库目录。"""
    name = Path(pdf_path).stem
    return str(Path("data") / "indexes" / name)


def main() -> None:
    if len(sys.argv) < 3:
        print("用法: python -m litagent.rag.demo <pdf路径> <问题...>")
        print("示例: python -m litagent.rag.demo data/paper.pdf 这篇论文的贡献是什么")
        sys.exit(1)

    pdf_path, question = sys.argv[1], " ".join(sys.argv[2:])
    persist_dir = _persist_dir(pdf_path)

    store_dir = Path(persist_dir)
    if not store_dir.exists() or not any(store_dir.iterdir()):
        store = build_index(pdf_path, persist_dir)
    else:
        print("复用已有索引 ...")
        store = VectorStore(persist_dir)

    answer, sources = ask(question, store)

    print("=" * 60)
    print(answer)
    print("=" * 60)
    print("参考来源：")
    for s in sources:
        print(f"  - 第 {s['page_no']} 页（相似度 {s['score']:.3f}）")


if __name__ == "__main__":
    main()