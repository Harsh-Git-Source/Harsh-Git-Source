from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def make_random_forest(random_state=42):
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"
    )

def make_xgboost(random_state=42):
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )
