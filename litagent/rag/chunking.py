"""Phase 1：文本切分。

把整页文本切成更小、更适合检索和喂给模型的块，块之间保留重叠以保住上下文。
每个块记住来源页码，用于回答时溯源。
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from litagent.ingest.pdf_parser import Page


@dataclass
class Chunk:
    """一个文本块，附带它来自哪一页。"""

    text: str
    page_no: int
    chunk_index: int  # 该块在本页内的序号，用于生成唯一 id


def split_pages(
    pages: list[Page],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """把每页文本切分成块。

    Args:
        pages: 抽取出的 PDF 页面。
        chunk_size: 每块最大字符数。
        chunk_overlap: 相邻块重叠的字符数，避免把语义从中间切断。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[Chunk] = []
    for page in pages:
        for idx, piece in enumerate(splitter.split_text(page.text)):
            chunks.append(Chunk(text=piece, page_no=page.page_no, chunk_index=idx))
    return chunks