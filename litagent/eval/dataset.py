"""Phase 4（扩展版）：检索评测集。

评测集按论文组织，每条样本是「问题 + 预期命中页码集合」。页码与 PDF 抽取后的
page_no 一致（从 1 开始），也即向量库里每个 chunk 的 page_no。

设计要点：
- **中英对照**：每个主题一条中文问题 + 一条英文问题，用于验证跨语言改写是否有效；
- **类型分层**：按问题类型打标签（概念 / 参数 / 表格 / 架构），便于分类型看检索表现；
- **多论文**：覆盖 bert.pdf 与 paper.pdf（T.pdf 是同一篇 Transformer 论文的别名）。

页码均来自对 PDF 抽取文本的逐页核对（见历史 commit 的标注过程），不是拍脑袋写的。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    """一条评测样本（已展开为单一语言）。"""

    question: str              # 用户问题（中文或英文）
    gt_pages: frozenset[int]   # 预期命中的页码集合
    category: str              # 类型标签：概念 / 参数 / 表格 / 架构
    note: str = ""             # 说明这条在测什么


@dataclass(frozen=True)
class Case:
    """一个评测主题，含中英两条问题，共享同一 ground truth。"""

    zh: str                    # 中文问题
    en: str                    # 英文问题
    gt_pages: frozenset[int]   # 预期命中页码集合
    category: str              # 类型标签
    note: str = ""             # 说明


# ---------------------------------------------------------------------------
# bert.pdf（BERT，16 页）
# ---------------------------------------------------------------------------
BERT_CASES: list[Case] = [
    Case("BERT 的核心贡献是什么", "What are the main contributions of BERT",
         frozenset({2}), "概念", "Introduction 的三条技术贡献"),
    Case("BERT_base 的层数、隐藏维度和参数量分别是多少",
         "How many layers, hidden dimensions and parameters does BERT_base have",
         frozenset({3}), "参数", "3.1 Model Architecture 的 BERT_BASE/BERT_LARGE 参数"),
    Case("BERT 的输入表示是如何构造的", "How is the BERT input representation constructed",
         frozenset({4}), "概念", "WordPiece / [CLS] / [SEP] / segment embeddings"),
    Case("BERT 的两个预训练任务是什么", "What are the two pre-training tasks of BERT",
         frozenset({4}), "概念", "Masked LM + Next Sentence Prediction"),
    Case("BERT 预训练用了哪些语料、规模多大",
         "What corpora and scale are used to pre-train BERT",
         frozenset({5}), "参数", "BooksCorpus (800M) + English Wikipedia"),
    Case("BERT 在 GLUE 基准上的表现如何", "How does BERT perform on the GLUE benchmark",
         frozenset({6}), "表格", "Table 1 的 GLUE Test 结果"),
    Case("BERT 在 SQuAD 1.1 上的 F1 分数是多少", "What is BERT's F1 score on SQuAD 1.1",
         frozenset({7}), "表格", "Table 2 的 SQuAD 1.1 EM/F1"),
    Case("去掉 Next Sentence Prediction 任务对 BERT 有什么影响",
         "What is the effect of removing the Next Sentence Prediction task on BERT",
         frozenset({8}), "表格", "Table 5 预训练任务消融（No NSP 行）"),
    Case("模型规模对 BERT 的下游任务效果有何影响",
         "How does model size affect BERT's downstream task accuracy",
         frozenset({8, 9}), "参数", "5.2 Effect of Model Size 及其结果表"),
    Case("BERT 微调时使用的学习率和 epoch 数是多少",
         "What learning rate and number of epochs are used for BERT fine-tuning",
         frozenset({14}), "参数", "附录 A.3 微调超参（lr 5e-5/3e-5/2e-5, epochs 2/3/4）"),
]

# ---------------------------------------------------------------------------
# paper.pdf / T.pdf（Attention Is All You Need，15 页）
# ---------------------------------------------------------------------------
TRANSFORMER_CASES: list[Case] = [
    Case("Transformer 相比 RNN/CNN 的核心动机是什么",
         "What is the core motivation of the Transformer compared to RNNs and CNNs",
         frozenset({2}), "概念", "1 Introduction：放弃循环、靠注意力并行"),
    Case("Transformer 的整体架构由哪些部分组成",
         "What are the main components of the Transformer architecture",
         frozenset({3}), "架构", "Figure 1 + Encoder/Decoder 堆叠"),
    Case("Transformer 的 Scaled Dot-Product Attention 是如何计算的",
         "How is Scaled Dot-Product Attention computed in the Transformer",
         frozenset({4}), "概念", "3.2.1 公式 softmax(QK^T/√dk)V"),
    Case("Transformer 的多头注意力机制是怎样工作的",
         "How does multi-head attention work in the Transformer",
         frozenset({4, 5}), "概念", "3.2.2 Multi-Head Attention（定义在 p4，机制/收益在 p5）"),
    Case("Transformer 里 Position-wise Feed-Forward 网络是什么",
         "What is the position-wise feed-forward network in the Transformer",
         frozenset({5}), "架构", "3.3 两层全连接 + ReLU"),
    Case("Transformer 的位置编码用的是什么函数",
         "What functions are used for positional encoding in the Transformer",
         frozenset({6}), "概念", "3.5 正弦/余弦位置编码"),
    Case("自注意力相比循环/卷积层在复杂度和计算路径上有什么优势",
         "What are the advantages of self-attention over recurrent and convolutional layers",
         frozenset({6}), "表格", "Table 1 复杂度 + 4 Why Self-Attention"),
    Case("Transformer 在 WMT 2014 英德翻译上的 BLEU 分数是多少",
         "What BLEU score does the Transformer achieve on WMT 2014 English-to-German translation",
         frozenset({8}), "表格", "Table 2 的 newstest2014 BLEU 28.4"),
    Case("Transformer 架构变体的消融实验得出了什么结论",
         "What do the ablation experiments on Transformer architecture variations show",
         frozenset({9}), "表格", "Table 3 Variations on the Transformer architecture"),
]

# 论文名 → 评测集 的映射；paper.pdf 与 T.pdf 是同一篇 Transformer 论文（别名）
PAPER_CASES: dict[str, list[Case]] = {
    "bert": BERT_CASES,
    "paper": TRANSFORMER_CASES,
    "t": TRANSFORMER_CASES,
    "transformer": TRANSFORMER_CASES,
}


def expand(cases: list[Case], langs: tuple[str, ...] = ("zh", "en")) -> list[Query]:
    """把 Case 列表展开成中英两条 Query。"""
    queries: list[Query] = []
    for c in cases:
        if "zh" in langs:
            queries.append(Query(c.zh, c.gt_pages, c.category, c.note))
        if "en" in langs:
            queries.append(Query(c.en, c.gt_pages, c.category, c.note))
    return queries


def queries_for(paper_name: str, langs: tuple[str, ...] = ("zh", "en")) -> list[Query]:
    """按论文名（stem，含别名）取对应评测集并展开为中英 Query。

    找不到匹配论文时抛出 KeyError，避免「张冠李戴」——用错了论文的评测集。
    """
    key = paper_name.lower()
    if key not in PAPER_CASES:
        raise KeyError(
            f"未找到论文「{paper_name}」的评测集；可用：{sorted(set(PAPER_CASES))}"
        )
    return expand(PAPER_CASES[key], langs)