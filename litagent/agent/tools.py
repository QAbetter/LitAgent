"""Phase 3：工具层——把 Phase 0/1/2 的能力封装成 Agent 可调用的工具。

设计：
- ToolBox：一篇论文的「已加载资源 + 工具方法」，资源按需加载、结果按需缓存；
- TOOLS：OpenAI function calling 格式的工具 JSON Schema 列表，供上层 Agent 使用。

每个工具的输出都是「文本 + 来源标记」（页码 / 图号 / 表号），
便于 Agent 在最终回答里溯源。
"""

from pathlib import Path

from litagent.ingest.figure_extractor import (
    DESCRIBE_PROMPT,
    TABLE_MARKDOWN_PROMPT,
    Figure,
    extract_figures,
    extract_tables,
)
from litagent.rag.hybrid import HybridRetriever
from litagent.rag.pipeline import build_index
from litagent.rag.vector_store import VectorStore
from litagent.vision import answer_image


class ToolBox:
    """把论文能力封装成可调用工具，并管理按需加载的资源与缓存。"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.stem = Path(pdf_path).stem
        self._store: VectorStore | None = None
        self._hybrid: HybridRetriever | None = None
        self._figures: dict[int, Figure] = {}
        self._tables: dict[int, Figure] = {}
        self._figure_desc_cache: dict[int, str] = {}
        self._table_md_cache: dict[int, str] = {}
        # 本轮问答中用到（需要前端展示）的图片路径，由 PaperAgent.ask_detail 收集
        self.used_images: list[str] = []

    # ---------- 资源懒加载 ----------

    @property
    def store(self) -> VectorStore:
        """文本向量库：已有索引就复用，否则现场建立。"""
        if self._store is None:
            index_dir = Path("data") / "indexes" / self.stem
            if index_dir.exists() and any(index_dir.iterdir()):
                self._store = VectorStore(str(index_dir))
            else:
                self._store = build_index(self.pdf_path, str(index_dir))
        return self._store

    @property
    def hybrid(self) -> HybridRetriever:
        """混合检索器：BM25 词面 + 向量语义，跨语言改写，随 store 懒加载。"""
        if self._hybrid is None:
            self._hybrid = HybridRetriever(self.store)
        return self._hybrid

    @property
    def figures(self) -> dict[int, Figure]:
        """论文里所有带图注的图，按编号索引。"""
        if not self._figures:
            out = str(Path("data") / "figures" / self.stem)
            self._figures = {f.number: f for f in extract_figures(self.pdf_path, out)}
        return self._figures

    @property
    def tables(self) -> dict[int, Figure]:
        """论文里所有带表注的表格，按编号索引。"""
        if not self._tables:
            out = str(Path("data") / "tables" / self.stem)
            self._tables = {t.number: t for t in extract_tables(self.pdf_path, out)}
        return self._tables

    # ---------- 工具实现 ----------

    def search_text(self, query: str, top_k: int = 4) -> str:
        """文本检索（混合检索），返回相关片段及其页码。"""
        results = self.hybrid.search(query, top_k=top_k)
        if not results:
            return "（未检索到相关内容）"
        parts = [
            f"[第 {r['page_no']} 页，检索分 {r['score']:.3f}]\n{r['text']}" for r in results
        ]
        return "\n\n---\n\n".join(parts)

    def list_figures(self) -> str:
        """列出论文所有图及其图注，帮模型定位图编号。"""
        if not self.figures:
            return "（这篇论文没有检测到带图注的图）"
        lines = [f"论文共 {len(self.figures)} 张图："]
        for n in sorted(self.figures):
            f = self.figures[n]
            lines.append(f"- Figure {n}（第 {f.page_no} 页）: {f.caption.splitlines()[0]}")
        return "\n".join(lines)

    def describe_figure(self, number: int) -> str:
        """对指定图调用视觉模型，返回图注 + 视觉描述。"""
        fig = self.figures.get(number)
        if fig is None:
            return f"（没有 Figure {number}，可用 list_figures 查看有哪些图）"
        self.used_images.extend(fig.image_paths)
        if number not in self._figure_desc_cache:
            descs = [answer_image(p, DESCRIBE_PROMPT) for p in fig.image_paths]
            self._figure_desc_cache[number] = "\n\n".join(descs)
        return (
            f"Figure {number}（第 {fig.page_no} 页）\n"
            f"图注: {fig.caption}\n"
            f"视觉描述:\n{self._figure_desc_cache[number]}"
        )

    def list_tables(self) -> str:
        """列出论文所有表格及其表注，帮模型定位表编号。"""
        if not self.tables:
            return "（这篇论文没有检测到带表注的表格）"
        lines = [f"论文共 {len(self.tables)} 张表格："]
        for n in sorted(self.tables):
            t = self.tables[n]
            lines.append(f"- Table {n}（第 {t.page_no} 页）: {t.caption.splitlines()[0]}")
        return "\n".join(lines)

    def lookup_table(self, number: int) -> str:
        """对指定表调用视觉模型，返回表注 + Markdown 数据。"""
        t = self.tables.get(number)
        if t is None:
            return f"（没有 Table {number}，可用 list_tables 查看有哪些表）"
        self.used_images.extend(t.image_paths)
        if number not in self._table_md_cache:
            mds = [answer_image(p, TABLE_MARKDOWN_PROMPT) for p in t.image_paths]
            self._table_md_cache[number] = "\n\n".join(mds)
        return (
            f"Table {number}（第 {t.page_no} 页）\n"
            f"表注: {t.caption}\n"
            f"表格内容（Markdown）:\n{self._table_md_cache[number]}"
        )

    def call(self, name: str, args: dict) -> str:
        """按函数名分发到对应工具方法。"""
        if name == "search_text":
            return self.search_text(args.get("query", ""), args.get("top_k", 4))
        if name == "list_figures":
            return self.list_figures()
        if name == "describe_figure":
            return self.describe_figure(int(args.get("number", 0)))
        if name == "list_tables":
            return self.list_tables()
        if name == "lookup_table":
            return self.lookup_table(int(args.get("number", 0)))
        return f"（未知工具: {name}）"


# OpenAI function calling 格式的工具 Schema（type=function 列表）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "按语义检索论文正文，返回最相关的文本片段及其页码。适合回答方法论、概念定义、结论等文字性问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词，尽量用英文术语。"},
                    "top_k": {"type": "integer", "description": "返回片段数量，默认 4。"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_figures",
            "description": "列出论文里所有带图注的图（含编号、页码、图注）。当不确定某张图的编号时先调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_figure",
            "description": "读取并理解指定编号的图，返回其图注和视觉描述。适合回答图表、坐标、趋势、对比等可视化问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer", "description": "图编号，如 1 表示 Figure 1。"},
                },
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "列出论文里所有带表注的表格（含编号、页码、表注）。当不确定某张表的编号时先调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_table",
            "description": "读取指定编号的表格，返回转好的 Markdown 数据。适合回答具体数值、指标对比等表格型问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer", "description": "表编号，如 4 表示 Table 4。"},
                },
                "required": ["number"],
            },
        },
    },
]