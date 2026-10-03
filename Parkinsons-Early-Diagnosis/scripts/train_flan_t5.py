"""
Training entry point for the FLAN-T5 classification experiment.

The supplied report describes:
- google/flan-t5-small
- mean-pooled encoder representation + dropout + binary classifier
- full end-to-end fine-tuning
- LoRA with rank 16 and 4-bit quantization

This script intentionally does not fabricate trained checkpoints. It prepares
the architecture and training entry point; the dataset and compute required
for reproducing the reported result must be supplied by the user.
"""
import argparse
import pandas as pd
from src.models.flan_t5 import FlanT5Classifier, build_tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "lora"], default="full")
    parser.add_argument("--model-name", default="google/flan-t5-small")
    args = parser.parse_args()

    print(f"Selected mode: {args.mode}")
    print(f"Base model: {args.model_name}")
    print("Architecture initialized below; dataset/training loop should be run on authorized PPMI data.")
    tokenizer = build_tokenizer(args.model_name)
    model = FlanT5Classifier(args.model_name)
    print(f"Tokenizer vocab: {len(tokenizer)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

if __name__ == "__main__":
    main()
