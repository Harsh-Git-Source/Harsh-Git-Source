from src.config import settings


def test_embedding_model_matches_project_design():
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
