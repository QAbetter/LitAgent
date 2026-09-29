"""集中配置：从 .env 读取智谱 API 相关配置。"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # 智谱 OpenAI 兼容接口
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    zhipu_api_key: str = ""

    # 视觉（多模态）模型：免费单图 glm-4v-flash，可换 glm-4.6v / glm-4v-plus-0111
    vision_model: str = "glm-4v-flash"

    # 文本模型（Phase 1 起使用）
    text_model: str = "glm-4-flash"

    # 向量化（Embedding）模型
    embedding_model: str = "embedding-3"


settings = Settings()