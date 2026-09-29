"""Phase 0：多模态视觉问答。

用智谱 OpenAI 兼容接口对图片进行描述 / 问答。
"""

import base64
from pathlib import Path

from config.settings import settings
from litagent.llm import get_client


def _to_image_url(ref: str) -> str:
    """把图片路径或 URL 统一转成 data URL / 原 URL。"""
    if ref.startswith(("http://", "https://", "data:")):
        return ref
    path = Path(ref)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {ref}")
    ext = path.suffix.lower().lstrip(".") or "png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/{ext};base64,{b64}"


def answer_image(image: str, question: str, model: str | None = None) -> str:
    """对单张图片进行描述 / 问答。

    Args:
        image: 本地图片路径或图片 URL。
        question: 需要向模型提出的问题。
        model: 可选，覆盖配置里的视觉模型。
    """
    image_url = _to_image_url(image)
    resp = get_client().chat.completions.create(
        model=model or settings.vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": question},
                ],
            }
        ],
    )
    return resp.choices[0].message.content