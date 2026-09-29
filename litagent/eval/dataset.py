"""Phase 4（扩展版）：检索评测集。

评测集按论文组织，每条样本是「问题 + 预期命中页码集合」。页码与 PDF 抽取后的
page_no 一致（从 1 开始），也即向量库里每个 chunk 的 page_no。

设计要点：
- **中英对照**：每个主题一条中文问题 + 一条英文问题，用于验证跨语言改写是否有效；
- **类型分层**：按问题类型打标签（概念 / 参数 / 表格 / 架构），便于分类型看检索表现；
- **多论文**：覆盖 10 篇经典论文（BERT / Transformer / ResNet / RAG / Transformer-XL /
  ViT / DDPM / LoRA / CLIP / GPT-3，paper.pdf 与 T.pdf 是同一篇 Transformer 论文的别名）。

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

# ---------------------------------------------------------------------------
# resnet.pdf（Deep Residual Learning，12 页）
# ---------------------------------------------------------------------------
RESNET_CASES: list[Case] = [
    Case("ResNet 的残差学习思想是什么", "What is the residual learning idea of ResNet",
         frozenset({2}), "概念", "Figure 2：F(x)+x 恒等 shortcut 残差块"),
    Case("ResNet 为什么能训练上百层的网络而缓解退化问题",
         "Why can ResNet train networks over 100 layers deep while mitigating degradation",
         frozenset({3}), "概念", "3. Deep Residual Learning：极深网络的收益讨论"),
    Case("ResNet 各版本（18/34/50/101/152 层）的网络结构配置",
         "What are the architecture configurations of ResNet-18/34/50/101/152",
         frozenset({5}), "表格", "Table 1 各层数 conv1~conv5 配置表"),
    Case("ResNet 在 ImageNet 上的 top-1 错误率相比 VGG 如何",
         "How does ResNet's top-1 error on ImageNet compare to VGG",
         frozenset({6}), "表格", "ImageNet 验证集 top-1/top-5 error 表（VGG-16 28.07）"),
    Case("ResNet 的 bottleneck 残差块是如何设计的",
         "How is the bottleneck residual block of ResNet designed",
         frozenset({7}), "架构", "1×1 降维→3×3→1×1 升维三层结构"),
    Case("残差网络相比 plain 网络的训练收敛表现有何不同",
         "How does the training convergence of residual networks differ from plain networks",
         frozenset({8}), "概念", "训练曲线：plain 加深后训练误差反而更高"),
    Case("ResNet 在目标检测任务（Faster R-CNN）上有什么改进",
         "What improvement does ResNet bring to object detection with Faster R-CNN",
         frozenset({10}), "架构", "4.2 Object Detection：替换 backbone 为 ResNet"),
    Case("ResNet 在 COCO 检测任务上的 mAP 提升多少",
         "How much does ResNet improve mAP on COCO detection",
         frozenset({11}), "表格", "COCO 检测 mAP 结果表"),
]

# ---------------------------------------------------------------------------
# rag.pdf（Retrieval-Augmented Generation，19 页）
# ---------------------------------------------------------------------------
RAG_CASES: list[Case] = [
    Case("RAG 要解决的核心问题是什么", "What is the core problem RAG aims to solve",
         frozenset({2}), "概念", "知识密集型任务：参数化记忆 + 非参数化检索"),
    Case("RAG 的整体架构由哪几部分组成",
         "What are the main components of the RAG architecture",
         frozenset({2}), "架构", "Query Encoder + MIPS 检索 + Generator 生成器"),
    Case("RAG 的生成器如何结合检索到的文档进行生成",
         "How does the RAG generator condition on retrieved documents",
         frozenset({3}), "概念", "基于输入 x 与检索文档 z 逐 token 生成"),
    Case("RAG 的端到端训练目标是什么",
         "What is the end-to-end training objective of RAG",
         frozenset({4}), "概念", "最大化 marginal negative log-likelihood"),
    Case("RAG 在 Open-Domain QA 测试集上的分数如何",
         "What are RAG's scores on Open-Domain QA test sets",
         frozenset({6}), "表格", "Table 1 Open-Domain QA Test Scores"),
    Case("RAG 生成内容在人类评估中的表现如何",
         "How does RAG perform in human assessments of generated text",
         frozenset({8}), "表格", "Table 4：Factuality / Specificity 人类评估"),
    Case("RAG 实验使用了哪些数据集、各有多少实例",
         "What datasets does RAG use and how many instances each",
         frozenset({19}), "表格", "Table 7：各数据集实例数统计"),
    Case("RAG 使用的检索方式是什么",
         "What retrieval method does RAG use",
         frozenset({2}), "架构", "Dense passage retrieval + MIPS 最大内积搜索"),
]

# ---------------------------------------------------------------------------
# transformerxl.pdf（Transformer-XL，20 页）
# ---------------------------------------------------------------------------
TRANSFORMERXL_CASES: list[Case] = [
    Case("Transformer-XL 要解决 Transformer 的什么问题",
         "What problem of vanilla Transformers does Transformer-XL address",
         frozenset({2}), "概念", "固定长度上下文导致的上下文碎片化"),
    Case("Transformer-XL 的段级循环机制是如何工作的",
         "How does the segment-level recurrence mechanism of Transformer-XL work",
         frozenset({2}), "概念", "复用上一段的 hidden states 而非从头计算"),
    Case("Transformer-XL 的相对位置编码是什么",
         "What is the relative positional encoding of Transformer-XL",
         frozenset({5}), "概念", "用相对距离 R 替代绝对位置编码"),
    Case("Transformer-XL 在训练和评估阶段的架构有何不同",
         "How does Transformer-XL's architecture differ between train and eval phases",
         frozenset({3}), "架构", "Train/Eval phase 架构图（段间状态传递）"),
    Case("Transformer-XL 的困惑度结果相比 LSTM/TCN 如何",
         "How do Transformer-XL's perplexity results compare to LSTM and TCN",
         frozenset({6}), "表格", "enwik8/text8 PPL 结果表"),
    Case("Transformer-XL 的消融实验说明了什么",
         "What do the ablation experiments of Transformer-XL show",
         frozenset({8}), "表格", "recurrence/encoding 消融（128M 27.02 PPL）"),
    Case("Transformer-XL 的结论是什么",
         "What is the conclusion of Transformer-XL",
         frozenset({9}), "概念", "5 Conclusions：建模更长依赖、评估更快"),
    Case("Transformer-XL 的注意力可视化展示了什么",
         "What does the attention visualization of Transformer-XL show",
         frozenset({15}), "架构", "长距离注意力可视化（1500+ token 依赖）"),
]

# ---------------------------------------------------------------------------
# vit.pdf（An Image is Worth 16x16 Words，22 页）
# ---------------------------------------------------------------------------
VIT_CASES: list[Case] = [
    Case("ViT 的核心思想是什么", "What is the core idea of ViT",
         frozenset({3}), "概念", "把图像切成 patch 当 token 输入 Transformer"),
    Case("ViT 相比 CNN 有什么不同",
         "How does ViT differ from CNNs",
         frozenset({2}), "概念", "不引入 translation equivariance/locality 归纳偏置"),
    Case("ViT 的整体架构由哪些部分组成",
         "What are the main components of the ViT architecture",
         frozenset({3}), "架构", "patch 线性投影 + Transformer Encoder + MLP Head"),
    Case("ViT-Base 的层数、隐藏维度和参数量是多少",
         "How many layers, hidden dimensions and parameters does ViT-Base have",
         frozenset({5}), "参数", "Model 表：ViT-Base 12 层 768 维 86M 参数"),
    Case("ViT 在 ImageNet 上相比 BiT 和 Noisy Student 的结果如何",
         "How does ViT compare to BiT and Noisy Student on ImageNet",
         frozenset({6}), "表格", "与 SOTA CNN 的对比结果表"),
    Case("ViT 在 CIFAR-10 上的精度是多少",
         "What accuracy does ViT achieve on CIFAR-10",
         frozenset({15}), "表格", "详细结果表（ViT-B/16 CIFAR-10 98.13）"),
    Case("ViT 的缩放研究得出了什么结论",
         "What conclusions does the scaling study of ViT draw",
         frozenset({8}), "概念", "4.4 Scaling Study：预训练算力与性能关系"),
    Case("ViT 学到的 embedding filters 有什么特点",
         "What are the characteristics of the embedding filters learned by ViT",
         frozenset({9}), "架构", "RGB embedding filters 类似 CNN 低层特征"),
]

# ---------------------------------------------------------------------------
# ddpm.pdf（Denoising Diffusion Probabilistic Models，25 页）
# ---------------------------------------------------------------------------
DDPM_CASES: list[Case] = [
    Case("扩散模型的基本思想是什么", "What is the basic idea of diffusion models",
         frozenset({2}), "概念", "前向逐步加噪、反向逐步去噪生成样本"),
    Case("扩散模型的图模型结构是怎样的",
         "What is the graphical model structure of diffusion models",
         frozenset({2}), "架构", "Figure 2：有向图模型 q(x_t|x_{t-1}) 与 p_θ(x_{t-1}|x_t)"),
    Case("扩散模型的前向过程和反向过程是如何定义的",
         "How are the forward and reverse processes of diffusion models defined",
         frozenset({2}), "概念", "前向固定高斯加噪、反向学习高斯去噪"),
    Case("扩散模型的训练算法流程是什么",
         "What is the training algorithm procedure of diffusion models",
         frozenset({4}), "概念", "Algorithm 1：采样 x0/t/ε，梯度下降噪声预测损失"),
    Case("扩散模型在 CIFAR10 上的结果如何",
         "What are the diffusion model's results on CIFAR10",
         frozenset({5}), "表格", "Table 1：CIFAR10 NLL/FID 对比"),
    Case("扩散模型在 LSUN 上生成样本的 FID 分数是多少",
         "What FID scores do diffusion models achieve on LSUN",
         frozenset({6}), "表格", "LSUN bedroom/church 等 FID 结果"),
    Case("扩散模型论文的结论是什么",
         "What is the conclusion of the diffusion model paper",
         frozenset({9}), "概念", "6 Conclusion：高质量合成、与其他方法建立联系"),
    Case("扩散模型如何从 x_t 估计原始图像 x_0",
         "How does a diffusion model estimate the original image x_0 from x_t",
         frozenset({7}), "概念", "4.2 渐进生成中基于部分信息估计 x_0"),
]

# ---------------------------------------------------------------------------
# lora.pdf（Low-Rank Adaptation，26 页）
# ---------------------------------------------------------------------------
LORA_CASES: list[Case] = [
    Case("LoRA 的核心思想是什么", "What is the core idea of LoRA",
         frozenset({3}), "概念", "冻结预训练权重，用低秩分解 ΔW=BA 建模更新"),
    Case("LoRA 相比全量微调有什么优势",
         "What advantages does LoRA have over full fine-tuning",
         frozenset({3}), "概念", "不增加推理延迟、可训练参数少、易切换任务"),
    Case("LoRA 的可训练参数量相比全量微调少了多少",
         "How many fewer trainable parameters does LoRA have compared to full fine-tuning",
         frozenset({6}), "参数", "GLUE 表 # Trainable Parameters 列对比"),
    Case("LoRA 在 GLUE 基准上的综合表现如何",
         "How does LoRA perform overall on the GLUE benchmark",
         frozenset({6}), "表格", "RoBERTa/DeBERTa GLUE 各任务分数表"),
    Case("LoRA 在 E2E NLG 挑战赛上的结果如何",
         "What are LoRA's results on the E2E NLG Challenge",
         frozenset({7}), "表格", "E2E NLG：BLEU/NIST/MET/ROUGE-L/CIDEr"),
    Case("LoRA 在 GPT-3 175B 上的微调结果如何",
         "What are LoRA's fine-tuning results on GPT-3 175B",
         frozenset({8}), "表格", "WikiSQL/MNLI-m/SAMSum 结果表"),
    Case("LoRA 的秩 r 应该取多大",
         "How large should the rank r of LoRA be",
         frozenset({9}), "概念", "7.1：很小的 r 已足够，验证低内在秩假设"),
    Case("LoRA 通常应用到哪些权重矩阵上",
         "Which weight matrices is LoRA typically applied to",
         frozenset({10}), "架构", "只作用注意力 Wq/Wv 性价比最高"),
]

# ---------------------------------------------------------------------------
# clip.pdf（Learning Transferable Visual Models From Natural Language，48 页）
# ---------------------------------------------------------------------------
CLIP_CASES: list[Case] = [
    Case("CLIP 的核心思想是什么", "What is the core idea of CLIP",
         frozenset({2}), "概念", "图文对比学习：从自然语言监督中学视觉表示"),
    Case("CLIP 相比传统监督学习有什么优势",
         "What advantages does CLIP have over traditional supervised learning",
         frozenset({6}), "概念", "zero-shot 迁移：无需任务特定训练数据"),
    Case("CLIP 的训练过程是怎样的",
         "What is the training procedure of CLIP",
         frozenset({5}), "架构", "伪代码：image/text encoder + 对比损失"),
    Case("CLIP 的 zero-shot 分类结果如何",
         "What are CLIP's zero-shot classification results",
         frozenset({7}), "表格", "zero-shot 与 linear probe 对比表（aYahoo/ImageNet/SUN）"),
    Case("CLIP 的 prompt engineering 起什么作用",
         "What role does prompt engineering play in CLIP",
         frozenset({8}), "概念", "用文本模板弥合训练与 zero-shot 分布差距"),
    Case("CLIP 的 few-shot 学习表现相比 BiT 如何",
         "How does CLIP's few-shot performance compare to BiT",
         frozenset({9}), "表格", "few-shot 线性分类器对比结果"),
    Case("CLIP 有哪些局限性",
         "What are the limitations of CLIP",
         frozenset({20}), "概念", "6 Limitations：数据效率低、难任务仍欠优"),
    Case("CLIP 在图像检索任务上的表现如何",
         "How does CLIP perform on image retrieval tasks",
         frozenset({46}), "表格", "Flickr30k/MSCOCO 检索结果表"),
]

# ---------------------------------------------------------------------------
# gpt3.pdf（Language Models are Few-Shot Learners，75 页）
# ---------------------------------------------------------------------------
GPT3_CASES: list[Case] = [
    Case("GPT-3 的核心思想是什么", "What is the core idea of GPT-3",
         frozenset({3}), "概念", "in-context learning：few-shot 无需梯度更新"),
    Case("GPT-3 各版本模型的规模是多少",
         "What are the sizes of the different GPT-3 model variants",
         frozenset({8}), "参数", "Model 表：125M（Small）到 175B 共 8 个版本"),
    Case("GPT-3 的模型架构有什么特点",
         "What are the characteristics of the GPT-3 model architecture",
         frozenset({8}), "架构", "2.1：与 GPT-2 同构的 dense transformer"),
    Case("GPT-3 在 LAMBADA、StoryCloze、HellaSwag 上的表现如何",
         "How does GPT-3 perform on LAMBADA, StoryCloze and HellaSwag",
         frozenset({12}), "表格", "3.1 常识/完形填空任务结果表"),
    Case("GPT-3 在 TriviaQA 等问答任务上相比 RAG 如何",
         "How does GPT-3 compare to RAG on QA tasks such as TriviaQA",
         frozenset({13}), "表格", "3.3 闭卷问答：TriviaQA/WebQS/NQ 结果表"),
    Case("GPT-3 训练过程中的数据去重对性能有何影响",
         "How does data deduplication during training affect GPT-3's performance",
         frozenset({31}), "概念", "4.1 训练曲线：去重与验证集重叠消融"),
    Case("GPT-3 如何过滤 Common Crawl 数据",
         "How does GPT-3 filter the Common Crawl data",
         frozenset({43}), "概念", "附录 A：按参考语料质量过滤 + 模糊去重"),
    Case("GPT-3 训练使用的算力规模有多大",
         "How much compute is used to train GPT-3",
         frozenset({46}), "参数", "附录 D：训练总算力统计"),
]

# 论文名 → 评测集 的映射；paper.pdf 与 T.pdf 是同一篇 Transformer 论文（别名）
PAPER_CASES: dict[str, list[Case]] = {
    "bert": BERT_CASES,
    "paper": TRANSFORMER_CASES,
    "t": TRANSFORMER_CASES,
    "transformer": TRANSFORMER_CASES,
    "resnet": RESNET_CASES,
    "rag": RAG_CASES,
    "transformerxl": TRANSFORMERXL_CASES,
    "vit": VIT_CASES,
    "ddpm": DDPM_CASES,
    "lora": LORA_CASES,
    "clip": CLIP_CASES,
    "gpt3": GPT3_CASES,
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