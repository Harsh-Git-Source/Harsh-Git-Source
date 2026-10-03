from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore

from src.config import settings
from src.embeddings.embedding_model import get_embedding_model


def build_retriever(
    summarized_items: list[tuple[Any, str]],
) -> MultiVectorRetriever:
    embeddings = get_embedding_model()

    # Start each ingestion session with a clean local collection so vectors
    # never outlive the in-memory parent document store.
    existing = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    try:
        existing.delete_collection()
    except Exception:
        pass

    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        search_kwargs={"k": settings.top_k},
    )

    parent_ids: list[str] = []
    summary_docs: list[Document] = []
    parent_docs: list[Document] = []

    for item, summary in summarized_items:
        parent_id = str(uuid.uuid4())
        parent_ids.append(parent_id)

        metadata = dict(item.metadata)
        metadata.update(
            {
                "modality": item.modality,
                "item_id": item.item_id,
                "image_path": item.image_path or "",
            }
        )

        summary_docs.append(
            Document(
                page_content=summary,
                metadata={**metadata, id_key: parent_id},
            )
        )

        parent_content = item.content if item.modality != "image" else (
            item.image_base64 or ""
        )
        parent_docs.append(
            Document(
                page_content=parent_content,
                metadata=metadata,
            )
        )

    if summary_docs:
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(parent_ids, parent_docs)))

    return retriever


def retrieve(
    retriever: MultiVectorRetriever,
    query: str,
) -> list[Document]:
    return retriever.invoke(query)


def reset_vectorstore() -> None:
    embeddings = get_embedding_model()
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    vectorstore.delete_collection()
