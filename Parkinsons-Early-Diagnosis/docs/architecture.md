# Architecture

## Stage 1 — Data preparation

1. Load authorized PPMI questionnaire CSVs.
2. Merge on patient/event identifiers where available.
3. Keep Screening-event records.
4. Remove columns above the missingness threshold.
5. Remove metadata/non-questionnaire fields.
6. Impute numeric questionnaire values.

## Stage 2 — Classical ML

- Random Forest benchmark
- XGBoost benchmark
- XGBoost feature importance
- Top-30 feature subset
- ANN benchmark using 30 → 128 → 64 → 32 → 1

## Stage 3 — LLM representation

Each questionnaire response can be transformed into:

`Question: <question>. Response: <answer>.`

The resulting patient-level text is tokenized using FLAN-T5's tokenizer.

## Stage 4 — FLAN-T5

The report describes:
- google/flan-t5-small
- mean pooling of encoder hidden states
- dropout 0.1
- binary classification head
- end-to-end fine-tuning
- LoRA rank 16 with 4-bit quantization

## Stage 5 — Interface

Streamlit provides a research prototype interface. Model checkpoints are intentionally not committed to Git because of size and reproducibility constraints.
