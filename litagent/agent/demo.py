"""Phase 3 演示入口：对一篇论文 PDF 提问，观察 Agent 的完整编排与溯源过程。

用法：
    python -m litagent.agent.demo <pdf路径>               # 交互式多轮（连续追问，验证多轮记忆）
    python -m litagent.agent.demo <pdf路径> <问题...>     # 单次问答

示例：
    python -m litagent.agent.demo data/bert.pdf
    python -m litagent.agent.demo data/bert.pdf BERT在SQuAD 1.1上的表现如何
"""

import sys

from litagent.agent import PaperAgent


def _print_answer(answer: str) -> None:
    print("=" * 60)
    print(answer)


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python -m litagent.agent.demo <pdf路径> [问题...]")
        print("      python -m litagent.agent.demo data/bert.pdf            # 交互式多轮")
        print("      python -m litagent.agent.demo data/bert.pdf BERT是什么  # 单次问答")
        sys.exit(1)

    pdf_path = sys.argv[1]
    agent = PaperAgent(pdf_path)

    # 带问题 -> 单次问答（向后兼容）
    if len(sys.argv) >= 3:
        _print_answer(agent.ask(" ".join(sys.argv[2:])))
        return

    # 不带问题 -> 交互式多轮，验证多轮记忆
    print("已进入多轮研读模式。输入问题回车；输入 exit / quit / q 或按 Ctrl+C 退出。")
    while True:
        try:
            question = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q", "退出"):
            break
        _print_answer(agent.ask(question))


if __name__ == "__main__":
    main()