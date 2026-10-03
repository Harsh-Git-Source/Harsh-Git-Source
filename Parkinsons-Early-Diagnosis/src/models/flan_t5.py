import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

class FlanT5Classifier(nn.Module):
    """
    FLAN-T5 encoder + mean pooling + dropout + binary classifier,
    matching the architecture described in the supplied report.
    """
    def __init__(self, model_name="google/flan-t5-small", dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.d_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.classifier(self.dropout(pooled))
        return logits.squeeze(-1)

def build_tokenizer(model_name="google/flan-t5-small"):
    return AutoTokenizer.from_pretrained(model_name)

def row_to_prompt(row, question_map=None):
    parts = []
    for col, value in row.items():
        if question_map and col in question_map:
            question = question_map[col]
        else:
            question = str(col).replace("_", " ")
        if value is None or str(value).strip() == "":
            continue
        parts.append(f"Question: {question}. Response: {value}.")
    return " ".join(parts)
