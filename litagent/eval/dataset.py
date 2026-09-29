"""Phase 4：检索评测集。

评测集用于量化检索质量，每条是一个「问题 + 预期命中页码集合」。
页码是 PDF 提取后的页码（从 1 开始），与向量库里每个 chunk 的 page_no 一致。

当前评测集针对 data/bert.pdf，覆盖两个已实证的 bug：
- BUG-1 跨语言：中文 query 检索英文正文（如「实验数据集」应命中数据集章节）。
- BUG-2 检索答偏：「核心贡献」应命中技术贡献，而非作者分工 / 模型参数等无关块。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    """一条评测样本。"""

    question: str              # 用户问题（中文或英文）
    gt_pages: frozenset[int]   # 预期命中的页码集合
    note: str = ""             # 说明这条在测什么


# bert.pdf 评测集
BERT_QUERIES: list[Query] = [
    Query(
        "实验数据集是什么",
        frozenset({7}),
        "跨语言：中文问题应命中数据集章节（SWAG 等）",
    ),
    Query(
        "What dataset is used for the experiments",
        frozenset({7}),
        "英文对照：同一问题的英文检索",
    ),
    Query(
        "BERT 的核心贡献是什么",
        frozenset({2}),
        "检索答偏：应命中 Introduction 技术贡献列表，而非模型参数",
    ),
    Query(
        "模型在 GLUE 测试集上的表现如何",
        frozenset({6}),
        "表格溯源：应命中 Table 1 的 GLUE Test 结果",
    ),
]