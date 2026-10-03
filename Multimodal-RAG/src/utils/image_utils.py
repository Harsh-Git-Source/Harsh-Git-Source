from __future__ import annotations

import base64
from pathlib import Path


def file_to_base64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def mime_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "application/octet-stream")
