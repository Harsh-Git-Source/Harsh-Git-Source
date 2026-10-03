from __future__ import annotations

import base64
from pathlib import Path

from google import genai
from google.genai import types

from src.config import settings
from src.utils.image_utils import mime_type


class GeminiMultimodal:
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Add it to your .env file."
            )
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    def summarize_image(self, image_path: str | Path) -> str:
        path = Path(image_path)
        image_bytes = path.read_bytes()
        # Keep the report's Base64-oriented design explicit. The current SDK
        # accepts image bytes as a multimodal content part.
        encoded = base64.b64encode(image_bytes)
        prompt = (
            "Describe this document image for retrieval. Capture charts, labels, "
            "numbers, entities, trends, relationships and visible text. "
            "Do not guess unreadable details."
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=base64.b64decode(encoded),
                    mime_type=mime_type(path),
                ),
            ],
        )
        return response.text.strip()

    def answer(
        self,
        question: str,
        text_context: list[str],
        image_paths: list[str],
    ) -> str:
        prompt = f"""
You are a multimodal document question-answering assistant.

Answer the user's question using ONLY the supplied document context.
If the context does not contain enough evidence, say so.

USER QUESTION:
{question}

TEXT / TABLE CONTEXT:
{chr(10).join(text_context) if text_context else "No text/table context retrieved."}

For images, inspect the supplied images directly.
Return a concise, evidence-grounded answer. Preserve exact numbers and units.
"""
        contents: list[object] = [prompt]
        for image_path in image_paths[:4]:
            path = Path(image_path)
            contents.append(
                types.Part.from_bytes(
                    data=path.read_bytes(),
                    mime_type=mime_type(path),
                )
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
        )
        return response.text.strip()
