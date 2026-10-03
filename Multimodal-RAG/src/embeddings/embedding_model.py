from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
