from pathlib import Path
import yaml
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from src.preprocessing.imputation import MissForestStyleImputer
from src.feature_engineering.feature_selection import select_top_features
from src.models.classical import make_random_forest, make_xgboost
from src.evaluation.metrics import classification_metrics

CONFIG = yaml.safe_load(open("config/config.yaml", "r", encoding="utf-8"))

def main():
    df = pd.read_csv("data/processed/screening_cleaned.csv")
    target = CONFIG["data"]["target_column"]

    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    X = df.drop(columns=[target])
    X = X.select_dtypes(include="number")
    X = X.loc[:, X.notna().mean() > 0]

    imputer = MissForestStyleImputer(random_state=CONFIG["project"]["seed"])
    X_imp = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=CONFIG["training"]["ml_test_size"],
        random_state=CONFIG["project"]["seed"], stratify=y
    )

    results = []

    for name, model in [
        ("Random Forest", make_random_forest()),
        ("XGBoost", make_xgboost())
    ]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        row = {"model": name, **classification_metrics(y_test, pred)}
        results.append(row)
        joblib.dump(model, f"models/{name.lower().replace(' ', '_')}.joblib")

    top_features, importance, cumulative = select_top_features(
        X_imp, y, CONFIG["data"]["top_k_features"]
    )
    pd.DataFrame({"feature": importance.index, "importance": importance.values,
                  "cumulative_importance": cumulative.values}).to_csv(
        "results/generated/feature_importance.csv", index=False
    )

    X30 = X_imp[top_features]
    Xt, Xv, yt, yv = train_test_split(
        X30, y, test_size=CONFIG["training"]["ml_test_size"],
        random_state=CONFIG["project"]["seed"], stratify=y
    )
    xgb30 = make_xgboost()
    xgb30.fit(Xt, yt)
    results.append({"model": "XGBoost Top 30", **classification_metrics(yv, xgb30.predict(Xv))})

    joblib.dump({"imputer": imputer, "features": list(X_imp.columns),
                 "top_features": top_features}, "models/preprocessing.joblib")
    joblib.dump(xgb30, "models/xgboost_top30.joblib")

    Path("results/generated").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv("results/generated/ml_metrics.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
