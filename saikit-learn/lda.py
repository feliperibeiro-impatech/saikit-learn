from discriminant_analysis import _BaseDiscriminant
import numpy as np

class LDA(_BaseDiscriminant):
    def _fit_covariance(self, X, Y):
        n_samples, n_features = X.shape
        n_classes = len(self._classes)

        scatter = np.zeros((n_features, n_features))
        for k, c in enumerate(self._classes):
            centro = X[Y == c] - self._means[k]
            scatter += centro.T @ centro

        self._covariance = scatter / (n_samples - n_classes)
