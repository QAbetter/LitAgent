"""Phase 2 顶层入口：多模态抽取 + 图文/表格对齐 + 视觉理解。

用法：
    python -m litagent.multimodal <pdf路径> [figures|tables|all]

做的事：
1. 从 PDF 抽取图片/表格并与图注/表注对齐（见 ingest/figure_extractor.py）；
2. 复用 Phase 0 的视觉模型：图 -> 文字描述，表格 -> Markdown 文本。

示例：
    python -m litagent.multimodal data/paper.pdf            # 只做图
    python -m litagent.multimodal data/bert.pdf tables      # 只做表格
    python -m litagent.multimodal data/bert.pdf all         # 图和表格都做
"""

import sys
from pathlib import Path

from litagent.ingest.figure_extractor import (
    DESCRIBE_PROMPT,
    TABLE_MARKDOWN_PROMPT,
    extract_figures,
    extract_tables,
)
from litagent.vision import answer_image


def describe_figures(pdf_path: str, out_dir: str) -> list[dict]:
    """抽取论文图片并对每张图生成视觉描述。

    Args:
        pdf_path: 本地 PDF 文件路径。
        out_dir: 图片落盘目录。

    Returns:
        每个子图一个 dict，含 number / page_no / caption / image / images / sub_index / description。
    """
    figures = extract_figures(pdf_path, out_dir)
    results: list[dict] = []
    for fig in figures:
        for idx, img_path in enumerate(fig.image_paths):
            description = answer_image(img_path, DESCRIBE_PROMPT)
            results.append(
                {
                    "number": fig.number,
                    "page_no": fig.page_no,
                    "caption": fig.caption,
                    "image": img_path,          # 当前子图
                    "images": fig.image_paths,  # 该 Figure 全部子图
                    "sub_index": idx,           # 子图序号，从 0 开始
                    "description": description,
                }
            )
    return results


def tables_to_markdown(pdf_path: str, out_dir: str) -> list[dict]:
    """抽取论文表格，并用视觉模型把每张表格截图转成 Markdown。

    Args:
        pdf_path: 本地 PDF 文件路径。
        out_dir: 表格截图落盘目录。

    Returns:
        每个表格一个 dict，含 number / page_no / caption / image / markdown。
    """
    tables = extract_tables(pdf_path, out_dir)
    results: list[dict] = []
    for t in tables:
        markdowns = [answer_image(img, TABLE_MARKDOWN_PROMPT) for img in t.image_paths]
        results.append(
            {
                "number": t.number,
                "page_no": t.page_no,
                "caption": t.caption,
                "image": t.image_paths[0],
                "markdown": "\n\n".join(markdowns),
            }
        )
    return results


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python -m litagent.multimodal <pdf路径> [figures|tables|all]")
        print("示例: python -m litagent.multimodal data/paper.pdf tables")
        sys.exit(1)

    pdf_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "figures"
    stem = Path(pdf_path).stem

    if mode in ("figures", "all"):
        out_dir = str(Path("data") / "figures" / stem)
        results = describe_figures(pdf_path, out_dir)
        if not results:
            print("未抽取出带图注的图片。")
        for r in results:
            print("=" * 60)
            sub = ""
            if len(r["images"]) > 1:
                sub = f" 子图 {r['sub_index'] + 1}/{len(r['images'])}"
            print(f"Figure {r['number']}{sub}（第 {r['page_no']} 页）")
            print(f"图注: {r['caption']}")
            print(f"图片: {r['image']}")
            print(f"描述: {r['description']}")

    if mode in ("tables", "all"):
        out_dir = str(Path("data") / "tables" / stem)
        results = tables_to_markdown(pdf_path, out_dir)
        if not results:
            print("未抽取出带表注的表格。")
        for r in results:
            print("=" * 60)
            print(f"Table {r['number']}（第 {r['page_no']} 页）")
            print(f"表注: {r['caption']}")
            print(f"截图: {r['image']}")
            print(f"Markdown:\n{r['markdown']}")


if __name__ == "__main__":
    main()