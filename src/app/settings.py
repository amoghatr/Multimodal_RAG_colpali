import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant settings
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str

    # Supabase settings
    supabase_url: str
    supabase_key: str

    # Anthropic settings
    anthropic_api_key: str

    # ColPali settings
    colpali_model_name: str = "vidore/colqwen2-v1.0"

    # Supabase bucket
    bucket: str = "colpali"


@lru_cache
def get_settings() -> Settings:
    return Settings()
