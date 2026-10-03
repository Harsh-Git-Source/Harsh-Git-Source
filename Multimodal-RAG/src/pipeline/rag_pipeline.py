from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.ingestion.pdf_parser import ContentItem, parse_pdf
from src.llm.gemini_client import GeminiMultimodal
from src.processing.summarizer import summarize_items
from src.retrieval.multivector_retriever import build_retriever, retrieve


class MultimodalRAGPipeline:
    def __init__(self) -> None:
        self.retriever = None
        self.gemini = None
        self.stats: dict[str, Any] = {
            "documents": 0,
            "text": 0,
            "table": 0,
            "image": 0,
        }

    def ingest(self, pdf_paths: list[str | Path]) -> dict[str, Any]:
        all_items: list[ContentItem] = []
        for path in pdf_paths:
            all_items.extend(parse_pdf(path))

        self.stats = {
            "documents": len(pdf_paths),
            "text": sum(i.modality == "text" for i in all_items),
            "table": sum(i.modality == "table" for i in all_items),
            "image": sum(i.modality == "image" for i in all_items),
        }

        summarized = summarize_items(all_items)
        self.retriever = build_retriever(summarized)
        self.gemini = GeminiMultimodal()

        return {
            **self.stats,
            "indexed_items": len(summarized),
        }

    def ask(self, question: str) -> dict[str, Any]:
        if not self.retriever or not self.gemini:
            raise RuntimeError("Process at least one PDF before asking a question.")

        retrieved = retrieve(self.retriever, question)

        text_context: list[str] = []
        image_paths: list[str] = []
        sources: list[dict[str, Any]] = []

        for doc in retrieved:
            modality = doc.metadata.get("modality", "text")
            if modality == "image":
                image_path = doc.metadata.get("image_path")
                if image_path and Path(image_path).exists():
                    image_paths.append(image_path)
            else:
                text_context.append(doc.page_content)

            sources.append(
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "modality": modality,
                    "item_id": doc.metadata.get("item_id"),
                }
            )

        answer = self.gemini.answer(
            question=question,
            text_context=text_context,
            image_paths=image_paths,
        )

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_text": text_context,
            "retrieved_images": image_paths,
        }
