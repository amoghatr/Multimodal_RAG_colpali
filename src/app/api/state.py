from anthropic import AsyncAnthropic
from app.settings import Settings
from qdrant_client import AsyncQdrantClient
from supabase.client import AsyncClient as SupabaseAsyncClient


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def create_supabase_client(settings: Settings) -> SupabaseAsyncClient:
    return SupabaseAsyncClient(
        supabase_key=settings.supabase_key,
        supabase_url=settings.supabase_url,
    )


def create_anthropic_client(settings: Settings) -> AsyncAnthropic:
    return AsyncAnthropic(api_key=settings.anthropic_api_key)
