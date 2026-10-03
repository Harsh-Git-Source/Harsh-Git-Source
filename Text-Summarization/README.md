# Abstractive Text Summarization — CNN/DailyMail

An end-to-end comparison of a classical **BiLSTM encoder-decoder with Luong attention** and a pretrained **Google FLAN-T5 model adapted with LoRA** for abstractive news summarization on the CNN/DailyMail dataset.

> The repository preserves the original project notebooks and report while adding a clean repository structure, configuration, reproducibility notes, and reusable entry points.

## Project Objective

Long news articles contain a large amount of information and can be time-consuming to read. The objective is to automatically generate concise summaries while preserving important information.

The project compares two approaches:

1. **BiLSTM Encoder-Decoder** — pretrained GloVe embeddings, bidirectional LSTM encoder, unidirectional LSTM decoder, Luong attention, teacher forcing during training, and beam search during inference.
2. **FLAN-T5 + LoRA** — pretrained Transformer encoder-decoder adapted to CNN/DailyMail with Low-Rank Adaptation, reducing the number of trainable parameters.

The report describes the project as an abstractive summarization study on CNN/DailyMail.

## Results

The reported comparison in the project report is:

| Model | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|---|---:|---:|---:|
| BiLSTM Encoder-Decoder | 0.20 | 0.015 | 0.14 |
| FLAN-T5 + LoRA | 0.36 | 0.16 | 0.255 |

These are the values in the report's results table.

The project report concludes that the FLAN-T5 + LoRA approach produced higher reported ROUGE scores and attributes the difference to Transformer self-attention, pretrained knowledge, and parameter-efficient adaptation.

## Dataset

The implementation uses the Hugging Face `cnn_dailymail` dataset with configuration `3.0.0`.

The FLAN-T5 notebook prototypes on 50,000 shuffled training examples, 10,000 validation examples, and 5,000 test examples. The BiLSTM notebook uses 50,000 shuffled training examples, 5,000 validation examples, and 100 test examples for its prototype/evaluation flow.

The project report describes CNN/DailyMail as containing news articles paired with human-written highlights. The report also states that its FLAN-T5 training setup uses about 287k training examples in the full dataset context.

## Approach 1 — BiLSTM + Luong Attention

### Preprocessing

- Lowercase text
- Expand contractions
- Keep alphabetic tokens and whitespace
- NLTK tokenization
- Build a vocabulary with special tokens: `<pad>`, `<unk>`, `<start>`, `<end>`
- Maximum vocabulary size: 50,000
- Article length: 600 tokens
- Summary length: 150 tokens
- Frozen 300-dimensional GloVe embeddings

The report describes the same high-level preprocessing, including fixed-length padding/truncation and pretrained GloVe embeddings.

### Architecture

```text
Article
   │
   ▼
Tokenization + Vocabulary
   │
   ▼
Frozen GloVe Embeddings
   │
   ▼
Bidirectional LSTM Encoder
   │
   ├──────────────► Encoder hidden states
   │                         │
   ▼                         ▼
Final states ───────► LSTM Decoder + Luong Attention
                              │
                              ▼
                       Generated summary
                              │
                              ▼
                         Beam Search
```

The report specifies a bidirectional encoder, unidirectional decoder, Luong attention, teacher forcing and beam search.

### Training

The notebook uses:

- Adam optimizer
- Learning rate: `1e-4`
- Batch size: `16`
- Epochs: `3` in the supplied BiLSTM notebook
- Teacher forcing ratio: `0.5`
- Gradient clipping at `1.0`
- StepLR scheduler
- NLL loss with padding ignored

The report gives a broader documented setup including hidden size 256 per direction and beam width 5.

## Approach 2 — FLAN-T5 + LoRA

### Preprocessing

- Clean article and highlights
- Use the pretrained T5 tokenizer
- Prefix each input with `summarize:`
- Maximum input length: 600 tokens in the supplied notebook
- Maximum target length: 150 tokens

The report describes the T5 text-to-text formulation and summarization prefix.

### LoRA

The supplied implementation loads `google/flan-t5-small` and applies PEFT LoRA to the sequence-to-sequence model with:

- rank `r=8`
- `lora_alpha=16`
- dropout `0.1`
- target modules `q` and `v`

The base model weights are frozen while the LoRA parameters are trained. The report describes this as parameter-efficient adaptation.

### Training

The supplied notebook uses:

- Batch size: `16`
- Learning rate: `5e-5`
- Epochs: `5`
- Linear warmup/scheduling
- AdamW
- Beam search with `num_beams=4` during evaluation

The report documents a similar low-learning-rate, small-batch LoRA fine-tuning strategy and explains that only LoRA parameters are updated.

## Evaluation

The project uses:

- **ROUGE-1** — unigram overlap
- **ROUGE-2** — bigram overlap
- **ROUGE-L** — longest-common-subsequence based similarity
- **BLEU** — n-gram precision metric

The report identifies ROUGE as the primary summarization metric and also reports BLEU.

## Repository Structure

```text
text-summarization-cnn-dailymail/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── notebooks/
│   ├── 01_flan_t5_lora.ipynb
│   └── 02_bilstm_luong_attention.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── metrics.py
│   └── config.py
├── configs/
│   └── project.yaml
├── results/
│   └── metrics.csv
├── scripts/
│   └── README.md
└── docs/
    └── project_report.pdf
```

## Reproducibility

1. Create a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run the notebooks in order when GPU resources are available.
4. The notebooks download the CNN/DailyMail dataset through Hugging Face Datasets.
5. The BiLSTM notebook downloads GloVe 6B 300d embeddings.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the notebooks

```bash
jupyter notebook notebooks/01_flan_t5_lora.ipynb
jupyter notebook notebooks/02_bilstm_luong_attention.ipynb
```

## Hardware / Compute Note

Both approaches are computationally expensive on CPU, especially the 50k-example training flows. A CUDA-capable GPU is recommended for reproducing the training runs. The FLAN-T5 notebook explicitly selects CUDA when available.

## Important Reproducibility Notes

The original notebooks are retained as the project's implementation record. They should be treated as the source implementation rather than silently replaced by a different experiment.

The repository intentionally does **not** include large generated model checkpoints, downloaded GloVe files, or the full CNN/DailyMail dataset. These artifacts should be generated locally and are excluded by `.gitignore`.

## Business Use Cases

The project report identifies potential applications including news aggregation and media monitoring, personalized news feeds, PR/reputation monitoring, internal corporate communications, and regulatory intelligence/policy tracking.

## Future Improvements

- Add a single configurable training CLI for both approaches.
- Add automated experiment tracking.
- Add ROUGE/BLEU evaluation to a common evaluation pipeline.
- Add example input/output summaries.
- Add model-card style documentation for each trained checkpoint.
- Add CI checks for Python syntax and repository hygiene.


## License

This repository contains project code and documentation. Dataset and pretrained model assets remain subject to their respective licenses and terms of use.
