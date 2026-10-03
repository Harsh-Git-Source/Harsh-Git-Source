"""Dataset and text preprocessing helpers used by the supplied notebooks."""
import re
import contractions
from datasets import load_dataset


def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = contractions.fix(text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_cnn_dailymail(config: str = "3.0.0"):
    return load_dataset("cnn_dailymail", config)


def sample_splits(dataset, train_size=50_000, validation_size=5_000, test_size=100, seed=42):
    train = dataset["train"].shuffle(seed=seed).select(range(min(train_size, len(dataset["train"]))))
    validation = dataset["validation"].shuffle(seed=seed).select(range(min(validation_size, len(dataset["validation"]))))
    test = dataset["test"].shuffle(seed=seed).select(range(min(test_size, len(dataset["test"]))))
    return train, validation, test
