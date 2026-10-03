import numpy as np
import pandas as pd
from xgboost import XGBClassifier

def select_top_features(X, y, top_k=30, random_state=42):
    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    top = importance.head(top_k)
    cumulative = importance.cumsum() / importance.sum()
    return list(top.index), importance, cumulative
