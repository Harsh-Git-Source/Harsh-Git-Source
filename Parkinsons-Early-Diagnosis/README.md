# Parkinson's Disease Early Screening — ML & FLAN-T5

An end-to-end research repository for **questionnaire-based early screening of Parkinson's Disease** using traditional machine learning, an ANN benchmark, and FLAN-T5 with LoRA / end-to-end fine-tuning.

> **Important:** This is a research/educational prototype, not a medical diagnostic device. The accuracy figures in this repository are the results reported in the supplied project report; they are not independently reproduced here.

## Project pipeline

```text
PPMI Online questionnaire data
        │
        ▼
Merge by Patient ID + Event ID
        │
        ▼
Keep Screening event
        │
        ▼
Remove >50% missing / metadata columns
        │
        ▼
Missing-value imputation
        │
        ▼
143 questionnaire features
        │
        ├───────────────► Random Forest
        │
        ├───────────────► XGBoost
        │                     │
        │                     ▼
        │                Top 30 features
        │                     │
        │                     ▼
        │                    ANN
        │
        ▼
Questionnaire → natural-language prompt
        │
        ▼
FLAN-T5 Small
   ├── LoRA + 4-bit quantization
   └── End-to-end fine-tuning
```

## Reported project results

| Model | Input | Reported accuracy |
|---|---|---:|
| Random Forest | 143 features | 91.56% |
| Random Forest | Top 30 | 90.23% |
| XGBoost | 143 features | 92.20% |
| XGBoost | Top 30 | 91.41% |
| ANN | Top 30 | 92.01% |
| FLAN-T5 + LoRA | 143 questions | 83.41% |
| FLAN-T5 + LoRA | Top 30 questions | 87.50% |
| FLAN-T5 end-to-end | Complete feature set | 94.49% |

These numbers are transcribed from the supplied report and should be treated as **reported experimental results**, not independently verified benchmarks.

## Dataset

The project report describes PPMI Online questionnaire data. The original workflow started with approximately 890 questionnaire columns, merged records using Patient ID and Event ID, filtered to the Screening event, removed columns with more than 50% missingness and irrelevant metadata, and reported a final dataset of 143 questionnaire columns and 38,256 records.

The dataset is **not included** in this repository because it is not part of the uploaded project files. Obtain the data through the appropriate PPMI access process and place local files under `data/raw/`.

## Repository structure

```text
parkinsons-early-diagnosis-ml-llm/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── config/
├── src/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── scripts/
├── notebooks/
├── data/
├── models/
├── results/
├── docs/
└── tests/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare data

Place the authorized PPMI CSV files in:

```text
data/raw/
```

Then run:

```bash
python scripts/prepare_data.py
```

The script creates a cleaned table under `data/processed/`.

## Train traditional ML models

```bash
python scripts/train_ml.py
```

This trains:

- Random Forest
- XGBoost
- Top-feature XGBoost selection
- ANN benchmark

Model artifacts are written to `models/` and metrics to `results/generated/`.

## FLAN-T5

The FLAN-T5 code provides the classification architecture used by the project:

```bash
python scripts/train_flan_t5.py --mode full
```

For LoRA:

```bash
python scripts/train_flan_t5.py --mode lora
```

Full end-to-end FLAN-T5 fine-tuning is computationally expensive and was described in the supplied report as the resource-intensive experiment. The repository therefore provides the reproducible training code rather than committing large model checkpoints.

## Streamlit interface

```bash
streamlit run app.py
```

The UI can run in **demo mode** without a trained checkpoint. When a trained model artifact is available, the app can load it for inference.

## Healthcare safety note

This application is intended for research, education, and demonstration. It must not be presented to users as a clinical diagnosis. Any real-world deployment would require clinical validation, appropriate governance, privacy controls, calibration, bias analysis, and regulatory review.

## Source artifacts

The supplied project report and presentation are preserved under `docs/`, and the uploaded classification notebook is preserved under `notebooks/` as the original source artifact.

## License

MIT.
