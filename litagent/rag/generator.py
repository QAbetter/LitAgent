"""Phase 1：生成器（把检索到的上下文喂给文本模型，生成带溯源的答案）。"""

from config.settings import settings
from litagent.llm import get_client


def _build_prompt(question: str, results: list[dict]) -> str:
    context = "\n\n".join(f"[来源页码 {r['page_no']}]\n{r['text']}" for r in results)
    return (
        "你是专业的文献研读助手。请只根据下面给出的论文片段回答问题，"
        "不要编造片段里没有的内容；如果资料不足以回答，请明确说明。\n\n"
        f"论文片段：\n{context}\n\n"
        f"问题：{question}\n"
        "回答（在关键结论处注明来源页码，如【第 N 页】）："
    )


def generate(question: str, results: list[dict]) -> str:
    """根据检索到的上下文生成最终答案。"""
    prompt = _build_prompt(question, results)
    resp = get_client().chat.completions.create(
        model=settings.text_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content