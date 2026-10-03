from __future__ import annotations

from src.ingestion.pdf_parser import ContentItem
from src.llm.gemini_client import GeminiMultimodal
from src.llm.groq_client import GroqSummarizer


def summarize_items(items: list[ContentItem]) -> list[tuple[ContentItem, str]]:
    groq = GroqSummarizer()
    gemini = GeminiMultimodal()
    output: list[tuple[ContentItem, str]] = []

    for item in items:
        if item.modality in {"text", "table"}:
            summary = groq.summarize(item.content, item.modality)
        elif item.modality == "image" and item.image_path:
            summary = gemini.summarize_image(item.image_path)
        else:
            continue
        output.append((item, summary))
    return output
