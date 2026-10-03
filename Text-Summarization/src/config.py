from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetConfig:
    name: str = "cnn_dailymail"
    version: str = "3.0.0"
    seed: int = 42

@dataclass(frozen=True)
class BiLSTMConfig:
    article_max_length: int = 600
    summary_max_length: int = 150
    vocabulary_size: int = 50_000
    min_frequency: int = 5
    embedding_dim: int = 300
    hidden_size: int = 256
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 3
    teacher_forcing_ratio: float = 0.5
    beam_width: int = 5

@dataclass(frozen=True)
class FlanT5LoRAConfig:
    model_name: str = "google/flan-t5-small"
    max_input_length: int = 600
    max_target_length: int = 150
    batch_size: int = 16
    learning_rate: float = 5e-5
    epochs: int = 5
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    beam_width: int = 4
