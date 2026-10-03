from pathlib import Path
import pandas as pd

def load_csv_files(raw_dir: str):
    raw_path = Path(raw_dir)
    files = sorted(raw_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_path}")
    return {p.stem: pd.read_csv(p) for p in files}

def merge_on_keys(frames: dict, left_keys, right_keys=None):
    """Merge questionnaire frames on available patient/event keys."""
    if not frames:
        raise ValueError("No dataframes supplied.")
    right_keys = right_keys or left_keys
    result = None
    for _, df in frames.items():
        if result is None:
            result = df.copy()
            continue
        lk = [c for c in left_keys if c in result.columns]
        rk = [c for c in right_keys if c in df.columns]
        if len(lk) == len(rk) and lk:
            result = result.merge(df, left_on=lk, right_on=rk, how="outer")
        else:
            common = [c for c in left_keys if c in result.columns and c in df.columns]
            if not common:
                continue
            result = result.merge(df, on=common, how="outer")
    return result
