"""Phase 3：论文研读 Agent——用 function calling 编排工具，多轮协作并溯源。

Agent 拿到用户问题后不预设流程，而是让语言模型自主决定：
1. 调用哪个工具（文本检索 / 看图 / 读表）；
2. 传什么参数；
3. 综合工具返回的证据，给出带「页码 / 图号 / 表号」溯源的最终答案。

核心是一个循环：模型输出 tool_call → 执行工具 → 结果喂回模型，直到模型不再调工具、直接作答。
"""

import json

from openai import OpenAI

from config.settings import settings
from litagent.agent.tools import TOOLS, ToolBox


SYSTEM_PROMPT = (
    "你是专业的学术论文研读助手，正在研读一篇已加载到本地的论文 PDF。"
    "你可以调用工具获取论文内容：\n"
    "- search_text：检索正文文字（适合方法论、概念、结论类问题）；\n"
    "- list_figures / describe_figure：查图清单 / 理解某张图；\n"
    "- list_tables / lookup_table：查表清单 / 读取某张表的数据。\n\n"
    "规则：\n"
    "1. 先判断问题需要文字、图还是表格，选用合适的工具，可以多轮调用组合多个工具。\n"
    "2. 要解读某张图/表但不确定编号时，先用 list_figures / list_tables 查编号。\n"
    "3. 严格基于工具返回的内容回答，不要编造论文里没有的信息；信息不足就明说。\n"
    "4. 最终回答必须在关键结论处标注来源，格式如【第 N 页】【Figure N】【Table N】。\n"
    "5. 你解读的图/表会以图片形式展示在回答下方，可以引用「上图 / 下图」指代；"
    "但关键结论仍要用文字说清楚，不要只写一句「如上图所示」而不解释。"
)


class PaperAgent:
    """一篇论文的研读 Agent。"""

    def __init__(self, pdf_path: str, max_steps: int = 8):
        self.box = ToolBox(pdf_path)
        self.max_steps = max_steps
        # 对话历史：system 起头，后续每轮问答的工具往返都累积在此，实现多轮记忆
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self) -> None:
        """清空对话历史，开始一个新话题。"""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _client(self) -> OpenAI:
        return OpenAI(api_key=settings.zhipu_api_key, base_url=settings.zhipu_base_url)

    def ask(self, question: str, verbose: bool = True) -> str:
        """回答用户问题，返回带溯源的最终答案（纯文本，向后兼容 CLI）。"""
        return self.ask_detail(question, verbose)["answer"]

    def ask_detail(self, question: str, verbose: bool = True) -> dict:
        """回答并返回结构化结果：{answer, images, tools}，供 Web 界面渲染图片与工具轨迹。"""
        self.box.used_images.clear()
        trace: list[dict] = []
        self.messages.append({"role": "user", "content": question})
        for _ in range(self.max_steps):
            resp = self._client().chat.completions.create(
                model=settings.text_model,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                final = msg.content or "（模型未给出回答）"
                self.messages.append({"role": "assistant", "content": final})
                return self._pack(final, trace)

            # 把 assistant 带 tool_calls 的消息回填（手动构造，避免 SDK 额外字段干扰）
            self.messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                if verbose:
                    print(f"[工具调用] {tc.function.name}({args})")
                trace.append({"name": tc.function.name, "arguments": args})
                result = self.box.call(tc.function.name, args)
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        overrun = "（达到最大工具调用步数，仍未完成回答，请换个问法）"
        self.messages.append({"role": "assistant", "content": overrun})
        return self._pack(overrun, trace)

    def _pack(self, answer: str, trace: list[dict]) -> dict:
        """组装结构化返回：图片去重保序。"""
        return {
            "answer": answer,
            "images": list(dict.fromkeys(self.box.used_images)),
            "tools": trace,
        }