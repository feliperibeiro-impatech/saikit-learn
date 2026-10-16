import pandas as pd
import numpy as np

class _BaseDiscriminant:
    def __init__(self, label: str):
        self._label = label
        self._classes = None
        self._feature_names = None
        self._priors = {}
        self._means = {}
        
    def fit(self, data: pd.DataFrame):
        self._feature_names = [col for col in data.columns if col != self._label]

        X = data[self._feature_names].to_numpy(dtype=np.float64)
        Y = data[self._label].to_numpy()

        self._classes = np.unique(Y)
        n_samples = len(Y)

        for c in self._classes:
            X_c = X[Y == c]

            self._priors[c] = len(X_c) / n_samples
            self._means[c] = np.mean(X_c, axis = 0)

        self._fit_covariance(X, Y)
        return self

    def _fit_covariance(self, X: np.ndarray, Y: np.ndarray):
        raise NotImplementedError("Implemente no LDA ou QDA")