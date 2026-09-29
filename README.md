# LitAgent —— 多模态文献研读 Agent

一个面向 AI 求职的作品集项目：把一篇本地 PDF 论文变成「能看图、会读表、可溯源、跨语言」的可对话研读助手。

区别于在网页版 AI（ChatGPT / 豆包）里贴论文：本项目的每一个回答都严格基于**本地论文的检索证据**，并在关键结论处标注**来源（页码 / 图号 / 表号）**；图表以真实图片形式展示，不会「凭空编造」，且整套流程（抽取 → 索引 → 检索 → 编排 → 评测）可复现、可度量。

## 核心亮点

- **多模态**：文本 + 图表（图/表渲染成图后由 `glm-4v-flash` 视觉模型理解）并重。
- **RAG 检索**：PyMuPDF 抽取 → 分块 → `embedding-3` 向量化 → ChromaDB 余弦检索。
- **混合检索**：BM25 词面匹配 + 向量语义检索，RRF 融合，修复跨语言与答偏问题。
- **跨语言查询改写**：中文问题先改写为英文检索词，跨越中英 embedding 语义鸿沟。
- **Agent 编排**：`glm-4-flash` function calling 多轮循环，自主选择 5 个工具并溯源。
- **评测闭环**：38 条评测集（2 篇论文 × 中英对照 × 类型分层）+ 检索指标（Hit Rate / Recall@k / MRR）+ 4 档消融实验。
- **离线可复现评测**：golden 快照把 embedding / LLM 改写 / 精排「录制 → 回放」，评测不联网、不烧钱、确定性。
- **可测试、可 CI**：`pytest` 离线单元+集成测试（注入确定性桩）+ GitHub Actions + ruff 静态检查。

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
    hybrid.py            # Phase 4：混合检索 + 跨语言改写 + LLM 精排（可配置消融）
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
    dataset.py           # 检索评测集（38 条，2 论文，中英对照 + 类型分层）
    metrics.py           # Hit Rate / Recall@k / MRR
    run_eval.py          # 4 档消融对比入口（支持 --offline 离线回放）
    golden.py            # golden 快照：录制/回放，让评测离线、确定性可复现
  llm.py                 # 统一 LLM 客户端（复用连接 + 超时/重试）
config/settings.py       # 配置（读取 .env）
tests/                   # pytest：metrics / chunking / 向量库 / 混合检索 / golden / agent 循环
.github/workflows/ci.yml # GitHub Actions：ruff + pytest + import 冒烟
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

### 检索评测（4 档消融）

```bash
python -m litagent.eval.run_eval data/bert.pdf     # BERT 论文（20 条）
python -m litagent.eval.run_eval data/paper.pdf    # Transformer 论文（18 条）
# 每个命令跑 4 档检索配置：纯向量 → +BM25 → +跨语言改写 → +LLM 精排，
# 输出逐条明细 + 汇总指标（Hit Rate / Recall@k / MRR）+ 分类型分层

# 离线、确定性回放（不联网、不烧钱，适合 CI/演示）：
python -m litagent.eval.golden --record data/bert.pdf   # 先联网录制一次快照
python -m litagent.eval.run_eval data/bert.pdf --offline # 之后可完全离线复现
```

## 测试与 CI

```bash
pip install -r requirements-dev.txt   # 额外安装 pytest / ruff
pytest -q                             # 全部离线、注入确定性桩，不联网
ruff check litagent tests config      # 静态检查
```

- **离线确定性**：测试与评测均不调用真实 API——向量、LLM 改写/精排用注入的确定性桩
  或 golden 快照回放替代，CI 上可复现。
- **CI**：`.github/workflows/ci.yml` 在 Ubuntu 多 Python 版本下跑 `ruff + pytest + import 冒烟`。
- **pre-commit**：`.pre-commit-config.yaml` 在提交前自动跑 ruff 与离线测试。

## 简历要点（工程化增量）

1. **统一 LLM 客户端**（`litagent/llm.py`）：复用连接 + 超时/重试，消除各处重复 `_client()`。
2. **检索器依赖注入**：`HybridRetriever` 的 embed / 改写 / 精排三个外部调用点可替换，
   是实现「离线复现评测」与单元测试的基础。
3. **golden 快照回归**：把外部调用「录制 → 回放」，评测从「每次联网烧钱」变成「离线确定性」，
   支持 CI 里做检索回归。

## 各阶段

- **Phase 0** 多模态视觉问答 —— 调通 `glm-4v-flash` 图片问答。
- **Phase 1** 文本 RAG —— PDF 抽取 → 切分 → 向量化 → 检索 → 生成。
- **Phase 2** 多模态抽取 —— 图/表渲染成图，视觉模型转 Markdown，图文对齐（处理双栏、公式位图、跨图注串扰等边界）。
- **Phase 3** Agent 编排 —— function calling 多轮循环 + 5 工具 + 溯源 + 多轮记忆；Web 界面 + 多论文管理。
- **Phase 4** 评测闭环 + 检索增强 —— 混合检索（BM25 + 向量 RRF）+ 跨语言改写 + LLM 精排；38 条评测集 + 4 档消融实验，量化各组件增量。

## 检索优化效果（消融实验）

评测集共 **38 条**（BERT 20 条 + Transformer 18 条），每条主题含中英两问，按类型分为概念 / 参数 / 表格 / 架构，`top_k=4`。四档检索配置逐级叠加，量化每个组件的增量：

| 配置 | 说明 |
| --- | --- |
| 纯向量 | `embedding-3` 向量检索（Phase 1 基线） |
| +BM25 | 词面 + 向量 RRF 融合（无改写） |
| +跨语言改写 | 中文问题先译成英文检索词，再做 BM25 + 向量混合 |
| +LLM 精排 | 召回后由 `glm-4-flash` 对候选逐条打分重排 |

**data/bert.pdf（20 条）**

| 指标 | 纯向量 | +BM25 | +跨语言改写 | +LLM 精排 |
| --- | --- | --- | --- | --- |
| Hit Rate | 0.800 | 0.700 | 0.850 | **0.900** |
| Recall@k | 0.750 | 0.700 | 0.850 | **0.900** |
| MRR | 0.629 | 0.579 | 0.642 | **0.825** |

**data/paper.pdf（Transformer，18 条）**

| 指标 | 纯向量 | +BM25 | +跨语言改写 | +LLM 精排 |
| --- | --- | --- | --- | --- |
| Hit Rate | 0.889 | 0.944 | **1.000** | 0.944 |
| Recall@k | 0.861 | 0.889 | **0.944** | 0.944 |
| MRR | 0.713 | 0.704 | 0.782 | **0.815** |

结论：

1. **跨语言改写是关键增量**：`embedding-3` 中英向量距离大，中文问题不改写就检索不到英文正文；改写后 BERT 的 Hit@4 从 0.80 提到 0.85+。
2. **BM25 离不开改写**：中文原文做英文词面匹配无效，甚至稀释向量结果（BERT 上 +BM25 反而降到 0.70）。生产路径因此是「改写 + BM25 + 向量」，而非单独加 BM25。
3. **LLM 精排提升排序质量**：让命中片段更靠前，MRR 在 BERT（0.642→0.825）、Transformer（0.782→0.815）上都有明显提升。

默认检索路径为「跨语言改写 + BM25 + 向量」（agent 的 `search_text` 所用）；LLM 精排作为可选升级，在消融里切换 `use_rerank=True` 即可开启。