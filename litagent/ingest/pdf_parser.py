"""Phase 1：PDF 文本抽取（带页码）。

用 PyMuPDF 把论文 PDF 逐页抽成纯文本，供后续切分、向量化使用。
"""

from dataclasses import dataclass

import pymupdf


@dataclass
class Page:
    """一页 PDF 抽取出的文本。"""

    page_no: int  # 页码，从 1 开始
    text: str


def extract_pdf(pdf_path: str) -> list[Page]:
    """抽取 PDF 全文，按页返回文本。

    Args:
        pdf_path: 本地 PDF 文件路径。

    Returns:
        每页一个 Page；空页会被跳过。
    """
    doc = pymupdf.open(pdf_path)
    pages: list[Page] = []
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append(Page(page_no=i + 1, text=text))
    finally:
        doc.close()
    return pages