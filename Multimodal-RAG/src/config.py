from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    gemini_api_key: str | None
    groq_model: str
    gemini_model: str
    embedding_model: str
    chroma_dir: Path
    upload_dir: Path
    image_dir: Path
    top_k: int
    collection_name: str
    parsing_strategy: str
    infer_table_structure: bool


def load_settings() -> Settings:
    config_path = ROOT_DIR / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}

    models = config.get("models", {})
    retrieval = config.get("retrieval", {})
    parsing = config.get("parsing", {})

    settings = Settings(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", models.get("groq", "llama-3.3-70b-versatile")),
        gemini_model=os.getenv("GEMINI_MODEL", models.get("gemini", "gemini-3.8-flash")),
        embedding_model=models.get(
            "embedding", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chroma_dir=ROOT_DIR / os.getenv("CHROMA_DIR", "data/chroma"),
        upload_dir=ROOT_DIR / os.getenv("UPLOAD_DIR", "data/uploads"),
        image_dir=ROOT_DIR / os.getenv("IMAGE_DIR", "data/extracted_images"),
        top_k=int(os.getenv("TOP_K", retrieval.get("top_k", 6))),
        collection_name=retrieval.get("collection_name", "multimodal_rag"),
        parsing_strategy=parsing.get("strategy", "hi_res"),
        infer_table_structure=bool(parsing.get("infer_table_structure", True)),
    )
    for directory in (settings.chroma_dir, settings.upload_dir, settings.image_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return settings


settings = load_settings()
