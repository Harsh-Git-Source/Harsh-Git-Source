from pathlib import Path

import pytest

from src.ingestion.pdf_parser import parse_pdf


def test_parser_requires_existing_file():
    with pytest.raises(Exception):
        parse_pdf(Path("does-not-exist.pdf"))
