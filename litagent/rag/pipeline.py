"""Phase 1：RAG 流水线，把「切分 → 向量化 → 检索 → 生成」四步串起来。"""

from litagent.ingest.pdf_parser import extract_pdf
from litagent.rag.chunking import split_pages
from litagent.rag.embedding import embed_texts
from litagent.rag.generator import generate
from litagent.rag.retriever import retrieve
from litagent.rag.vector_store import VectorStore


def build_index(
    pdf_path: str,
    persist_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> VectorStore:
    """对一篇 PDF 建立索引：抽取 → 切块 → 向量化 → 写入向量库。

    Returns:
        已建好的 VectorStore，可直接用于提问。
    """
    print("[1/4] 抽取 PDF 文本 ...")
    pages = extract_pdf(pdf_path)
    print(f"      共 {len(pages)} 页有文本")

    print("[2/4] 切分文本块 ...")
    chunks = split_pages(pages, chunk_size, chunk_overlap)
    print(f"      共 {len(chunks)} 个块")

    print("[3/4] 向量化（embedding-3）...")
    embeddings = embed_texts([c.text for c in chunks])
    print(f"      完成，向量维度 {len(embeddings[0])}")

    print("[4/4] 写入向量库 ...")
    store = VectorStore(persist_dir)
    store.add_chunks(chunks, embeddings)
    print(f"      已写入 {store.count()} 条")
    return store


def ask(question: str, store: VectorStore, top_k: int = 4) -> tuple[str, list[dict]]:
    """检索 + 生成，返回 (最终答案, 参考来源列表)。"""
    results = retrieve(question, store, top_k=top_k)
    answer = generate(question, results)
    return answer, results