"""统一的 LLM 客户端封装，消除各模块重复的 `_client()`。

集中管理连接（复用同一个 OpenAI 客户端实例）、超时与重试。
vision / embedding / generator / agent / hybrid 都从这里取客户端。

额外的价值：评测与测试可以替换 `chat*` 函数来离线回放 golden 快照，
使得整个 RAG + 检索管线在 CI 上不联网、确定性可复现。
"""

from openai import OpenAI

from config.settings import settings

# 复用的客户端实例（OpenAI 客户端是线程安全的，可多个请求共享）
_client: OpenAI | None = None


def get_client() -> OpenAI:
    """获取（懒加载、复用）OpenAI 兼容客户端实例。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.zhipu_api_key,
            base_url=settings.zhipu_base_url,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )
    return _client