"""Phase 3：Web 界面后端——多论文管理 + PaperAgent HTTP 接口。

启动：
    python -m litagent.ui.server
    浏览器访问 http://127.0.0.1:8000

功能：
- 扫描 data/ 下的所有 PDF 作为可选论文；
- 上传新论文（保存到 data/，自动处理同名）；
- 选择某篇论文后提问，Agent 回答中用到图/表会以图片形式返回给前端渲染。
"""

import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from litagent.agent import PaperAgent

# 项目根目录（data/ 与 litagent/ 同级），用 __file__ 计算，不依赖启动时 CWD
ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)  # 保证 ToolBox 里的相对路径 data/... 正确

DATA_DIR = ROOT / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="NewLitAgent 文献研读")

# 论文 id(stem) -> PDF 绝对路径；懒加载的 PaperAgent 缓存
_pdf_files: dict[str, Path] = {}
_agents: dict[str, PaperAgent] = {}


def _scan_papers() -> dict[str, Path]:
    """扫描 data/ 下所有 PDF，返回 {文件 stem: 绝对路径}。"""
    return {p.stem: p for p in sorted(DATA_DIR.glob("*.pdf"))}


def _get_agent(paper_id: str) -> PaperAgent:
    """按论文 id 取（懒加载）对应 Agent。"""
    if paper_id not in _pdf_files:
        raise HTTPException(status_code=404, detail=f"论文不存在: {paper_id}")
    if paper_id not in _agents:
        _agents[paper_id] = PaperAgent(str(_pdf_files[paper_id]))
    return _agents[paper_id]


def _to_url(path: str) -> str:
    """把本地图片路径转成前端可访问的相对 URL。"""
    posix = Path(path).as_posix()
    for prefix, url in (("data/figures/", "/files/figures/"), ("data/tables/", "/files/tables/")):
        if posix.startswith(prefix):
            return url + posix[len(prefix):]
    return posix


class AskRequest(BaseModel):
    paper_id: str
    question: str


class ResetRequest(BaseModel):
    paper_id: str


@app.get("/api/papers")
def list_papers() -> dict:
    """返回当前可选论文列表。"""
    return {"papers": [{"id": sid, "name": sid} for sid in sorted(_pdf_files)]}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:  # noqa: B008  # FastAPI 依赖注入惯用法
    """接收上传的 PDF，保存到 data/ 后返回新论文 id。"""
    raw = file.filename or "paper.pdf"
    if not raw.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持上传 PDF 文件")

    stem = Path(raw).stem
    # 同名（Windows 大小写不敏感）自动加时间戳后缀，避免覆盖已有论文
    existing = {p.name.lower() for p in DATA_DIR.glob("*.pdf")}
    if f"{stem}.pdf".lower() in existing:
        stem = f"{stem}_{int(time.time())}"
    dest = DATA_DIR / f"{stem}.pdf"

    dest.write_bytes(await file.read())
    _pdf_files[stem] = dest
    return {"id": stem, "name": stem}


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    """对指定论文提问，返回 answer + 图片 URL + 工具轨迹。"""
    result = _get_agent(req.paper_id).ask_detail(req.question)
    return {
        "answer": result["answer"],
        "images": [_to_url(p) for p in result["images"]],
        "tools": result["tools"],
    }


@app.post("/api/reset")
def reset(req: ResetRequest) -> dict:
    """清空指定论文的多轮对话历史。"""
    _get_agent(req.paper_id).reset()
    return {"ok": True}


# 静态资源：图片目录 + 前端页面
app.mount("/files/figures", StaticFiles(directory=str(ROOT / "data" / "figures")), name="figures")
app.mount("/files/tables", StaticFiles(directory=str(ROOT / "data" / "tables")), name="tables")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")

# 初始化论文列表
_pdf_files = _scan_papers()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)