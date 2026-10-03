import numpy as np
import pandas as pd
from src.feature_engineering.feature_selection import select_top_features

def test_select_top_features():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(100, 40)), columns=[f"f{i}" for i in range(40)])
    y = (X["f0"] + X["f1"] > 0).astype(int)
    top, importance, cumulative = select_top_features(X, y, top_k=5)
    assert len(top) == 5
    assert len(importance) == 40
