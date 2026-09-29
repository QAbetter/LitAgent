"""test_agent.py：PaperAgent function calling 多轮循环的单测（注入剧本式 fake client）。

验证：
  1) 工具被调用时能回填 tool_calls 消息并执行真实分发（mock ToolBox）；
  2) 模型不再调工具时返回最终答案；
  3) 返回结构含 answer / images / tools 且溯源信息在 messages 中累积。
"""

from litagent.agent.agent import PaperAgent


class _FakeMsg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChoices:
    def __init__(self, msg):
        self.message = msg


class _FakeResp:
    def __init__(self, msg):
        self.choices = [_FakeChoices(msg)]


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, messages, tools, tool_choice, **kwargs):
        return _FakeResp(self._responses.pop(0))


class _FakeChat:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)


class _FakeClient:
    def __init__(self, responses):
        self.chat = _FakeChat(responses)
        # 记录每次调用传入的 messages，供断言历史累积
        self.seen = []

    def _track(self, messages):
        self.seen.append(list(messages))


class _FakeBox:
    """替换 ToolBox：call 返回固定文本，used_images 记录本轮图片。"""

    def __init__(self):
        self.used_images = []

    def call(self, name, args):
        if name == "search_text":
            self.used_images.append("fig1.png")
            return "【第 4 页】BERT uses Masked LM"
        if name == "lookup_table":
            return "【Table 2】SQuAD F1: 93.2"
        return "（空）"


def _tool_call(tcid, name, arguments):
    tc = _FakeMsg()  # 占位：下面直接构造 tool_calls
    tc.id = tcid
    tc.function = type("F", (), {"name": name, "arguments": arguments})()
    return tc


def _build_agent(scripted_messages):
    box = _FakeBox()
    agent = PaperAgent("data/fake.pdf")
    agent.box = box
    responses = []
    for m in scripted_messages:
        tcs = m.get("tool_calls", []) or []
        if tcs:
            responses.append(
                _FakeMsg(content="", tool_calls=[_tool_call(t["id"], t["name"], t["args"]) for t in tcs])
            )
        else:
            responses.append(_FakeMsg(content=m.get("content", "")))
    client = _FakeClient(responses)
    agent._client = lambda: client  # 复用同一实例：响应按调用顺序推进
    return agent, box


def test_toolloop_then_final_answer():
    # 第 1 步：模型要求 search_text；第 2 步：直接给出最终答案
    agent, box = _build_agent([
        {"tool_calls": [{"id": "c1", "name": "search_text", "args": '{"query":"BERT pretrain"}'}]},
        {"content": "答案是 BERT uses Masked LM【第 4 页】。"},
    ])
    result = agent.ask_detail("BERT 预训练任务是什么？", verbose=False)

    assert result["answer"] == "答案是 BERT uses Masked LM【第 4 页】。"
    assert result["tools"] == [{"name": "search_text", "arguments": {"query": "BERT pretrain"}}]
    # 用到的图片被 URL 收集
    assert "fig1.png" in result["images"]
    # 多轮历史累积：system + user + assistant(tool_calls) + tool 结果 + assistant(最终)
    assert any(m["role"] == "tool" for m in agent.messages)


def test_does_not_return_images_unless_used():
    # 未用到任何图片型工具时，images 为空
    class _NoImgBox:
        used_images = []

        def call(self, name, args):
            return "纯文本检索结果"

    agent, _ = _build_agent([
        {"tool_calls": [{"id": "c1", "name": "lookup_table", "args": '{"number": 2}'}]},
        {"content": "完成。"},
    ])
    agent.box = _NoImgBox()
    result = agent.ask_detail("读表", verbose=False)
    assert result["images"] == []