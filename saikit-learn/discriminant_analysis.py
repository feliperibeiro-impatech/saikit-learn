import numpy as np

class _BaseDiscriminant:
    def __init__(self):
        self._classes = None
        self._priors = None   # shape (K,)
        self._means = None    # shape (K, d)
        self._n_features = None
        self._covariance = None

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y)

        if X.ndim != 2:
            raise ValueError("X deve ser 2D, com shape (n_samples, n_features).")
        if Y.ndim != 1 or len(Y) != X.shape[0]:
            raise ValueError("Y deve ser 1D e ter o mesmo número de amostras que X.")
        
        if np.isnan(X).any():
            raise ValueError("X contém NaN.")

        n_samples, self._n_features = X.shape
        self._classes = np.unique(Y)
        n_classes = len(self._classes)

        if n_samples <= n_classes:
            raise ValueError("É preciso ter mais amostras do que classes.")

        self._priors = np.zeros(n_classes)
        self._means = np.zeros((n_classes, self._n_features))

        for k, c in enumerate(self._classes):
            X_c = X[Y == c]
            self._priors[k] = len(X_c) / n_samples
            self._means[k] = X_c.mean(axis=0)

        self._fit_covariance(X, Y)
        return self

    def _fit_covariance(self, X: np.ndarray, Y: np.ndarray):
        raise NotImplementedError("Implemente no LDA ou QDA")