"""命令行入口。

用法：python -m litagent.cli <图片路径或URL> <问题...>
"""

import sys

from litagent.vision import answer_image


def main() -> None:
    if len(sys.argv) < 3:
        print("用法: python -m litagent.cli <图片路径或URL> <问题...>")
        print("示例: python -m litagent.cli test.png 这张图描述了什么方法")
        sys.exit(1)

    image, question = sys.argv[1], " ".join(sys.argv[2:])
    print("调用模型中...")
    answer = answer_image(image, question)
    print("=" * 48)
    print(answer)


if __name__ == "__main__":
    main()