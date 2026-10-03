from __future__ import annotations

from groq import Groq

from src.config import settings


class GroqSummarizer:
    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add it to your .env file."
            )
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def summarize(self, content: str, modality: str) -> str:
        label = "table" if modality == "table" else "document text"
        prompt = f"""
You are a retrieval-oriented document analyst.

Summarize the following {label}. Preserve:
- key entities and names
- important numbers, dates, percentages and units
- relationships between facts
- headings or categories when useful
- table row/column meaning if this is a table

Write a concise factual summary optimized for semantic retrieval.
Do not invent information.

CONTENT:
{content}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You produce factual retrieval summaries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=900,
        )
        return response.choices[0].message.content.strip()
