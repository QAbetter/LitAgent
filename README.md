# LitAgent —— 多模态文献研读 Agent

一个面向 AI 求职的作品集项目：把一篇本地 PDF 论文变成「能看图、会读表、可溯源、跨语言」的可对话研读助手。

区别于在网页版 AI（ChatGPT / 豆包）里贴论文：本项目的每一个回答都严格基于**本地论文的检索证据**，并在关键结论处标注**来源（页码 / 图号 / 表号）**；图表以真实图片形式展示，不会「凭空编造」，且整套流程（抽取 → 索引 → 检索 → 编排 → 评测）可复现、可度量。

## 核心亮点

- **多模态**：文本 + 图表（图/表渲染成图后由 `glm-4v-flash` 视觉模型理解）并重。
- **RAG 检索**：PyMuPDF 抽取 → 分块 → `embedding-3` 向量化 → ChromaDB 余弦检索。
- **混合检索**：BM25 词面匹配 + 向量语义检索，RRF 融合，修复跨语言与答偏问题。
- **跨语言查询改写**：中文问题先改写为英文检索词，跨越中英 embedding 语义鸿沟。
- **Agent 编排**：`glm-4-flash` function calling 多轮循环，自主选择 5 个工具并溯源。
- **评测闭环**：评测集 + 检索指标（Hit Rate / Recall@k / MRR），可量化对比检索方案。

## 技术栈

| 层 | 选型 |
| --- | --- |
| 接口 | 智谱 OpenAI 兼容 API（`https://open.bigmodel.cn/api/paas/v4/`） |
| 文本模型 | `glm-4-flash` |
| 视觉模型 | `glm-4v-flash`（免费单图，可换 `glm-4.6v` / `glm-4v-plus-0111`） |
| 向量模型 | `embedding-3`（2048 维） |
| 向量库 | ChromaDB（余弦相似度） |
| PDF 抽取 | PyMuPDF |
| 切分 | langchain-text-splitters |
| 词面检索 | rank-bm25 |
| Web 界面 | FastAPI + uvicorn + 原生 HTML/JS + marked.js |

## 目录结构

```
litagent/
  vision.py              # Phase 0：图片视觉问答
  cli.py                 # Phase 0：命令行入口
  ingest/
    pdf_parser.py        # Phase 1：PDF 文本抽取（带页码）
    figure_extractor.py  # Phase 2：图/表抽取与图文对齐
  rag/
    chunking.py          # 文本切分
    embedding.py         # 向量化（embedding-3，自动分批）
    vector_store.py      # ChromaDB 封装
    retriever.py         # 纯向量检索（baseline）
    hybrid.py            # Phase 4：混合检索 + 跨语言改写
    generator.py         # 带溯源的答案生成
    pipeline.py          # 建索引 + 问答流水线
  agent/
    tools.py             # 5 个工具 + function calling Schema
    agent.py             # PaperAgent：多轮编排 + 溯源
    demo.py              # 命令行演示入口
  multmodal.py           # Phase 2：图/表 → Markdown 一键入口
  ui/
    server.py            # FastAPI 后端（多论文上传/选择）
    static/index.html    # Web 聊天界面（真图展示 + 工具徽章）
  eval/
    dataset.py           # 检索评测集
    metrics.py           # Hit Rate / Recall@k / MRR
    run_eval.py          # 检索方案对比入口
config/settings.py       # 配置（读取 .env）
```

## 快速开始

环境：Windows + Python 3.10。

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置密钥：复制 .env 并填入智谱 API Key
#    ZHIPU_API_KEY=你的密钥

# 4. 把论文 PDF 放入 data/ 目录（例如 data/bert.pdf）
```

### 命令行研读（Agent 编排）

```bash
python -m litagent.agent.demo data/bert.pdf               # 交互式多轮问答
python -m litagent.agent.demo data/bert.pdf BERT是什么     # 单次问答
```

### Web 界面（多论文上传 + 真图展示）

```bash
python -m litagent.ui.server
# 浏览器打开 http://127.0.0.1:8000
```

### 检索评测

```bash
python -m litagent.eval.run_eval data/bert.pdf
# 输出 Hit Rate / Recall@k / MRR，对比「纯向量」vs「混合检索」
```

## 各阶段

- **Phase 0** 多模态视觉问答 —— 调通 `glm-4v-flash` 图片问答。
- **Phase 1** 文本 RAG —— PDF 抽取 → 切分 → 向量化 → 检索 → 生成。
- **Phase 2** 多模态抽取 —— 图/表渲染成图，视觉模型转 Markdown，图文对齐（处理双栏、公式位图、跨图注串扰等边界）。
- **Phase 3** Agent 编排 —— function calling 多轮循环 + 5 工具 + 溯源 + 多轮记忆；Web 界面 + 多论文管理。
- **Phase 4** 评测闭环 + 检索增强 —— 混合检索（BM25 + 向量 RRF）+ 跨语言查询改写，用指标量化修复效果。

## 检索优化效果（data/bert.pdf）

| 指标 | 纯向量检索 | 混合检索（改写+BM25+向量） |
| --- | --- | --- |
| Hit Rate | 0.250 | **1.000** |
| Recall@k | 0.250 | **1.000** |
| MRR | 0.083 | **0.875** |

修复了两个已知问题：

1. **中英跨语言语义鸿沟**：中文「实验数据集是什么」原相似度仅 0.35 检索失败，改写为英文检索词后正常命中数据集章节。
2. **检索答偏**：中文「核心贡献」原误命中作者分工/模型参数，混合检索后命中 Introduction 技术贡献列表。