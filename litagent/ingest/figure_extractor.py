"""Phase 2：从论文 PDF 抽取图片，并完成「图注（caption）↔ 图片」对齐。

思路：
1. 逐页用 `get_text("blocks")` 找到形如 "Figure 1: ..." 的图注文本块；
2. 图片有两种形态：
   - 嵌入位图：`get_images(full=True)` + `extract_image` 直接抽出；
   - 矢量图形：用 `get_drawings()` 找包围盒 + `get_pixmap` 渲染成 PNG；
3. 同页内，图片 bbox 底部 y1 < 图注顶部 y0 的图片，归给该图注。
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# 图注匹配：只 `match` 行首的 "Figure N:" / "Figure N."，避免误命中正文里的
# "Figure 1 shows ..." 这类引用（用 search 会误判）。
_CAPTION_RE = re.compile(r"^Figure\s+(\d+)\s*[:.]")

# 表注匹配：行首的 "Table N:" / "Table N."。
_TABLE_RE = re.compile(r"^Table\s+(\d+)\s*[:.]")

# 页面里所有图/表注（Figure 与 Table），用于按图注位置给页面分区，
# 避免上一张表格被卷进当前图的矢量渲染。
_CAPTION_ANY_RE = re.compile(r"^(?:Figure|Table)\s+\d+\s*[:.]")

# 位图被认作「图」的最小高度（pt）。正文里的公式、特殊符号常被 PDF 导出成
# 矮条位图（高十几 pt），真正带图注的图高度一般远超此值，据此过滤误判。
MIN_FIGURE_HEIGHT = 50.0

DESCRIBE_PROMPT = "请详细描述这张图展示的内容，包括坐标轴、图例、关键结果等。"

TABLE_MARKDOWN_PROMPT = (
    "请把这张表格完整转成 Markdown 表格格式（表头一行 + 数据行），"
    "逐格保留，数字和文字必须与图中一致，不要遗漏任何行列。"
)


@dataclass
class Figure:
    """一篇论文里的一张图（可能有多个子图，记作多张图片）。"""

    number: int  # 图编号，如 Figure 1 -> 1
    page_no: int  # 图片所在页码，从 1 开始
    caption: str  # 图注全文
    image_paths: list[str] = field(default_factory=list)  # 关联到的本地图片路径


@dataclass
class _Caption:
    """页面内定位到的一条图注。"""

    number: int
    text: str
    top: float  # 文本块顶部 y 坐标
    x0: float  # 文本块左边界
    x1: float  # 文本块右边界


@dataclass
class _Image:
    """页面内抽取出的一张嵌入图片。"""

    xref: int  # PDF 内部对象引用号
    path: str  # 落盘后的本地路径
    bottom: float  # 图片 bbox 底部 y 坐标
    height: float  # 图片 bbox 高度（pt），用于过滤正文里的公式位图
    x0: float  # 图片 bbox 左边界
    x1: float  # 图片 bbox 右边界


def extract_figures(pdf_path: str, out_dir: str) -> list[Figure]:
    """抽取 PDF 中的图片并与图注对齐。

    Args:
        pdf_path: 本地 PDF 文件路径。
        out_dir: 图片落盘目录（不存在会自动创建）。

    Returns:
        按图编号排序的 Figure 列表。
    """
    return _extract_captioned(pdf_path, out_dir, _CAPTION_RE)


def extract_tables(pdf_path: str, out_dir: str) -> list[Figure]:
    """抽取 PDF 中的表格并与表注对齐（结构同 Figure，image_paths 指向表格截图）。

    Args:
        pdf_path: 本地 PDF 文件路径。
        out_dir: 表格截图落盘目录（不存在会自动创建）。

    Returns:
        按表编号排序的 Figure 列表（字段含义同上）。
    """
    return _extract_captioned(pdf_path, out_dir, _TABLE_RE)


def _extract_captioned(pdf_path: str, out_dir: str, caption_re) -> list[Figure]:
    """抽取带图注内容的通用实现：图 / 表共用。

    caption_re: 用于匹配图注或表注的正则。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    figures: list[Figure] = []
    try:
        for page_index, page in enumerate(doc):
            page_no = page_index + 1
            captions = _find_captions(page, caption_re)
            if not captions:
                continue
            images = _extract_page_images(doc, page, page_no, out)
            drawings = page.get_drawings()  # 矢量图形，用于兜底渲染
            text_rects = [
                (b[0], b[1], b[2], b[3])
                for b in page.get_text("blocks")
                if b[6] == 0
            ]
            caption_positions = _caption_positions(page)  # 页面所有图/表注位置，用于分区
            vec_index = 0
            for cap in captions:
                lower_bound = _lower_bound_for(cap, caption_positions)
                linked = [
                    img.path
                    for img in images
                    if img.bottom < cap.top
                    and img.bottom >= lower_bound
                    and img.height >= MIN_FIGURE_HEIGHT
                    and _x_overlap((img.x0, img.x1), (cap.x0, cap.x1))
                ]
                if linked:
                    # 命中嵌入位图：直接关联「图注正上方」的图片
                    figures.append(
                        Figure(number=cap.number, page_no=page_no, caption=cap.text, image_paths=linked)
                    )
                else:
                    # 图是矢量绘制的（无嵌入位图）：把图注上方的矢量区域渲染成 PNG
                    bbox = _vector_bbox(drawings, cap, text_rects, lower_bound)
                    if bbox is None:
                        continue
                    rel_path = Path(f"p{page_no}_v{vec_index}.png")
                    _render_vector(page, bbox, out / rel_path)
                    vec_index += 1
                    figures.append(
                        Figure(number=cap.number, page_no=page_no, caption=cap.text,
                               image_paths=[str(out / rel_path)])
                    )
    finally:
        doc.close()

    figures.sort(key=lambda f: f.number)
    return figures


def _find_captions(page, caption_re=_CAPTION_RE) -> list[_Caption]:
    """从一页的文本块里找出所有图注/表注。"""
    captions: list[_Caption] = []
    for block in page.get_text("blocks"):
        # block 形如 (x0, y0, x1, y1, text, block_no, block_type)；type 0 为文本
        x0, y0, x1, y1, text, _block_no, block_type = block
        if block_type != 0:
            continue
        text = text.strip()
        m = caption_re.match(text)
        if m:
            captions.append(_Caption(number=int(m.group(1)), text=text, top=y0, x0=x0, x1=x1))
    return captions


def _x_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """判断两个横向区间 [x0, x1] 是否有重叠（用于区分双栏页面里的左右两栏）。"""
    return a[0] < b[1] and b[0] < a[1]


def _caption_positions(page) -> list[tuple[float, float, float, float]]:
    """返回页面里所有图/表注的 (x0, x1, top, bottom)，按 y 从小到大排序。

    用于把页面按图注位置分区（含横向列），避免上一张表格/图的内容被卷进当前图的渲染区。
    """
    positions: list[tuple[float, float, float, float]] = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text, _block_no, block_type = block
        if block_type != 0:
            continue
        if _CAPTION_ANY_RE.match(text.strip()):
            positions.append((x0, x1, y0, y1))
    positions.sort(key=lambda p: p[2])
    return positions


def _lower_bound_for(cap: _Caption, caption_positions) -> float:
    """计算某图注所属图区域的下界：同栏（横向重叠）内、它上方最近一条图/表注的底部。"""
    lower_bound = 0.0
    for x0, x1, top, bottom in caption_positions:
        if top < cap.top and _x_overlap((x0, x1), (cap.x0, cap.x1)):
            lower_bound = max(lower_bound, bottom)
    return lower_bound


def _extract_page_images(doc, page, page_no: int, out: Path) -> list[_Image]:
    """抽取一页里的嵌入位图到本地（按 xref 去重）。

    只落盘「高度达标」的位图：正文里的公式、符号常被导出成矮条位图，
    那些不落盘，避免污染输出目录。
    """
    images: list[_Image] = []
    seen: set[int] = set()

    for item in page.get_images(full=True):
        xref = item[0]
        if xref in seen:
            continue
        seen.add(xref)

        rects = page.get_image_rects(xref)
        if not rects:
            continue
        bottom = rects[0].y1
        height = rects[0].y1 - rects[0].y0
        if height < MIN_FIGURE_HEIGHT:
            continue  # 公式/图标等矮条位图，跳过

        try:
            raw = doc.extract_image(xref)
        except Exception:
            # 某些对象可能无法抽取（损坏 / 非位图），跳过不影响整体。
            continue

        ext = raw.get("ext", "png")
        rel_path = Path(f"p{page_no}_x{xref}.{ext}")
        (out / rel_path).write_bytes(raw["image"])

        images.append(
            _Image(
                xref=xref, path=str(out / rel_path), bottom=bottom, height=height,
                x0=rects[0].x0, x1=rects[0].x1,
            )
        )

    return images


def _vector_bbox(drawings, cap: _Caption, text_rects=None, lower_bound: float = 0.0):
    """计算图注正上方的矢量图形包围盒。

    把图注同栏（横向重叠）内、y1 < cap.top 且 y0 >= lower_bound 的所有矢量图形
    取并集；但矢量图里的文字是 text（而非 drawing），若只框图形会裁掉图内文字
    （尤其是图底部的标签）。所以再把图注上方、同栏、纵向下沿不低于图顶部的
    文字块一并纳入下边界。

    lower_bound: 图区域的下界（同栏内上一个图/表注的底部），避免卷进上一张表格。
    text_rects: `page.get_text("blocks")` 里文本块的矩形列表，每项 (x0, y0, x1, y1)。
    """
    rects = [
        d["rect"]
        for d in drawings
        if d["rect"].y1 < cap.top
        and d["rect"].y0 >= lower_bound
        and _x_overlap((d["rect"].x0, d["rect"].x1), (cap.x0, cap.x1))
    ]
    if not rects:
        return None
    x0 = min(r.x0 for r in rects)
    x1 = max(r.x1 for r in rects)
    y0 = min(r.y0 for r in rects)
    y1 = max(r.y1 for r in rects)

    if text_rects:
        for tx0, ty0, tx1, ty1 in text_rects:
            # 图内文字：在图注上方、同栏、且不高于图顶（留 3pt 容差）。
            below_caption = ty0 < cap.top
            in_x_range = _x_overlap((tx0, tx1), (cap.x0, cap.x1))
            not_above_figure = ty0 >= y0 - 3
            if below_caption and in_x_range and not_above_figure:
                y1 = max(y1, ty1)

    return pymupdf.Rect(x0 - 2, y0 - 2, x1 + 2, y1 + 2)


def _render_vector(page, bbox, out_path: Path) -> None:
    """把页面某块矢量区域渲染成 PNG 并落盘。"""
    pix = page.get_pixmap(clip=bbox, dpi=150)
    pix.save(str(out_path))