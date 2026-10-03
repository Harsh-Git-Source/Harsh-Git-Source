from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unstructured.partition.pdf import partition_pdf

from src.config import settings
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ContentItem:
    item_id: str
    modality: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    image_path: str | None = None
    image_base64: str | None = None


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def parse_pdf(pdf_path: str | Path) -> list[ContentItem]:
    pdf_path = Path(pdf_path)
    document_id = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:12]
    output_dir = settings.image_dir / _safe_name(pdf_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Parsing %s", pdf_path.name)

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy=settings.parsing_strategy,
        infer_table_structure=settings.infer_table_structure,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )

    items: list[ContentItem] = []

    for idx, element in enumerate(elements):
        category = type(element).__name__
        text = str(element).strip()
        metadata = dict(getattr(element, "metadata", {}) or {})

        page_number = metadata.get("page_number")
        base_metadata = {
            "document_id": document_id,
            "source": pdf_path.name,
            "page": page_number,
            "element_type": category,
        }

        if category == "Image":
            image_b64 = getattr(metadata, "image_base64", None) or metadata.get(
                "image_base64"
            )
            if not image_b64:
                continue
            try:
                image_bytes = base64.b64decode(image_b64)
                image_path = output_dir / f"image_{idx}.png"
                image_path.write_bytes(image_bytes)
            except Exception as exc:
                logger.warning("Could not save extracted image %s: %s", idx, exc)
                continue

            items.append(
                ContentItem(
                    item_id=f"{document_id}-image-{idx}",
                    modality="image",
                    content="",
                    metadata=base_metadata,
                    image_path=str(image_path),
                    image_base64=image_b64,
                )
            )
            continue

        if category == "Table":
            table_html = metadata.get("text_as_html") or metadata.get("table_as_html")
            content = table_html or text
            if content:
                items.append(
                    ContentItem(
                        item_id=f"{document_id}-table-{idx}",
                        modality="table",
                        content=content,
                        metadata=base_metadata,
                    )
                )
            continue

        if category in {"NarrativeText", "Title", "ListItem", "Header", "Footer"}:
            if text and len(text.strip()) >= 20:
                items.append(
                    ContentItem(
                        item_id=f"{document_id}-text-{idx}",
                        modality="text",
                        content=text,
                        metadata=base_metadata,
                    )
                )

    logger.info(
        "Extracted %d items from %s",
        len(items),
        pdf_path.name,
    )
    return items
