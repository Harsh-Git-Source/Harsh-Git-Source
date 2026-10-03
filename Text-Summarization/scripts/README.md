# Scripts

The original training and evaluation implementations are preserved in `notebooks/` because the supplied project was notebook-driven.

For a clean reproduction:

1. Install dependencies with `pip install -r requirements.txt`.
2. Run the FLAN-T5 notebook or BiLSTM notebook in a GPU-enabled environment.
3. Do not commit generated model checkpoints or downloaded GloVe/data artifacts; `.gitignore` excludes them.

The `src/` package contains reusable preprocessing, configuration, and evaluation helpers extracted from the project workflow.
