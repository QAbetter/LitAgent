"""test_chunking.py：文本切分单测（不联网）。

验证分块的数量、页码归属、以及长文本引发的多块行为。
"""

from litagent.ingest.pdf_parser import Page
from litagent.rag.chunking import split_pages


def _page(text: str, page_no: int) -> Page:
    return Page(page_no=page_no, text=text)


def test_short_page_single_chunk():
    chunks = split_pages([_page("hello world", 3)], chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].page_no == 3
    assert chunks[0].chunk_index == 0


def test_long_page_multiple_chunks_with_page_keeping():
    text = "word " * 2000  # 10000 字符 -> 超过一个块
    chunks = split_pages([_page(text, 7)], chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(c.page_no == 7 for c in chunks)
    # chunk_index 在本页内递增
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_indices_reset_per_page():
    text = "word " * 2000
    chunks = split_pages(
        [_page(text, 1), _page(text, 2)], chunk_size=500, chunk_overlap=50
    )
    first_idxs = [c.chunk_index for c in chunks if c.page_no == 1]
    second_idxs = [c.chunk_index for c in chunks if c.page_no == 2]
    assert first_idxs == list(range(len(first_idxs)))
    assert second_idxs == list(range(len(second_idxs)))


def test_chunks_are_nonempty():
    text = "a " * 500
    chunks = split_pages([_page(text, 1)], chunk_size=100, chunk_overlap=20)
    assert chunks
    assert all(c.text for c in chunks)