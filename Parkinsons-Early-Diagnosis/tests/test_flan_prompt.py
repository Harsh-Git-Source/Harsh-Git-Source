from src.models.flan_t5 import row_to_prompt

def test_row_to_prompt():
    prompt = row_to_prompt({"Q1": "Yes", "Q2": "No"}, {"Q1": "Do you have tremor?"})
    assert "Do you have tremor?" in prompt
    assert "Yes" in prompt
