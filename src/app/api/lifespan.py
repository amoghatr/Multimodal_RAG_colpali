from contextlib import asynccontextmanager
from typing import AsyncIterator, TypedDict

import instructor
from app.api.state import (
    create_anthropic_client,
    create_qdrant_client,
    create_supabase_client,
)
from app.colpali.loaders import ColQwen2Loader
from app.services.img_downloader import SupabaseJPEGDownloader
from app.services.img_uploader import SupabaseJPEGUploader
from app.settings import get_settings

# from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.models import ColQwen2, ColQwen2Processor
from fastapi import FastAPI
from instructor import AsyncInstructor
from qdrant_client import AsyncQdrantClient


class State(TypedDict):
    model: ColQwen2
    processor: ColQwen2Processor
    supabase_uploader: SupabaseJPEGUploader
    supabase_downloader: SupabaseJPEGDownloader
    instructor_client: AsyncInstructor
    qdrant_client: AsyncQdrantClient
    collection_name: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[State]:
    settings = get_settings()
    qdrant_client = create_qdrant_client(settings=settings)
    anthropic_client = create_anthropic_client(settings=settings)
    instructor_client = instructor.from_anthropic(client=anthropic_client)
    supabase_client = create_supabase_client(settings=settings)
    supabase_uploader = SupabaseJPEGUploader(
        client=supabase_client, bucket_name=settings.bucket
    )
    supabase_downloader = SupabaseJPEGDownloader(
        client=supabase_client, bucket_name=settings.bucket
    )
    loader = ColQwen2Loader(model_name=settings.colpali_model_name)
    model, processor = loader.load()

    yield {
        "model": model,
        "processor": processor,
        "supabase_uploader": supabase_uploader,
        "supabase_downloader": supabase_downloader,
        "instructor_client": instructor_client,
        "qdrant_client": qdrant_client,
        "collection_name": settings.collection_name,
    }

    await qdrant_client.close()
    await anthropic_client.close()
