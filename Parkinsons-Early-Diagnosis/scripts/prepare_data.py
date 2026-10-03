from pathlib import Path
import pandas as pd
import yaml

from src.preprocessing.data_loader import load_csv_files, merge_on_keys

CONFIG = yaml.safe_load(open("config/config.yaml", "r", encoding="utf-8"))

def main():
    frames = load_csv_files(CONFIG["data"]["raw_dir"])
    merged = merge_on_keys(
        frames,
        CONFIG["data"]["patient_id_columns"],
        CONFIG["data"]["event_id_columns"]
    )

    screening_values = {str(x).lower() for x in CONFIG["data"]["screening_values"]}
    event_candidates = [c for c in merged.columns if "event" in c.lower()]
    if event_candidates:
        col = event_candidates[0]
        mask = merged[col].astype(str).str.lower().isin(screening_values)
        if mask.any():
            merged = merged.loc[mask].copy()

    threshold = CONFIG["data"]["missing_threshold"]
    missing_ratio = merged.isna().mean()
    keep = missing_ratio[missing_ratio <= threshold].index
    cleaned = merged.loc[:, keep].copy()

    target = CONFIG["data"]["target_column"]
    if target not in cleaned.columns:
        raise KeyError(
            f"Target column {target!r} was not found. "
            f"Available columns include: {list(cleaned.columns)[:30]}"
        )

    out = Path(CONFIG["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out / "screening_cleaned.csv", index=False)
    print(f"Saved {cleaned.shape} to {out / 'screening_cleaned.csv'}")

if __name__ == "__main__":
    main()
