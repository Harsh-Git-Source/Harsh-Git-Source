import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

class MissForestStyleImputer:
    """
    Random-forest iterative imputation.

    The project report used MissForest. This implementation uses sklearn's
    IterativeImputer with a RandomForestRegressor when the original MissForest
    package is unavailable, keeping the repository portable on modern Python.
    """

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter
        self.imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=50,
                random_state=random_state,
                n_jobs=-1
            ),
            max_iter=max_iter,
            random_state=random_state,
            initial_strategy="median"
        )

    def fit_transform(self, X):
        X = pd.DataFrame(X).copy()
        numeric = X.select_dtypes(include=[np.number])
        if numeric.shape[1] != X.shape[1]:
            raise ValueError("Imputation stage expects numeric questionnaire features.")
        out = self.imputer.fit_transform(X)
        return pd.DataFrame(out, columns=X.columns, index=X.index)

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = self.imputer.transform(X)
        return pd.DataFrame(out, columns=X.columns, index=X.index)
