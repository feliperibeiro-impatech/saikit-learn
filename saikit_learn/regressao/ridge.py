from saikit_learn.utilidades.validacao import verify_array, verify_X_y, NotFittedError
import numpy as np

class Ridge:
    def __init__(self, lmbd: float = 1.0, intercepto : bool = True):
        self.lmbd = lmbd
        self.intercepto = intercepto
        self._fitted = False

    def fit(self, X:np.ndarray, y:np.ndarray):
        if not np.isfinite(self.lmbd) or self.lmbd < 0:
            raise ValueError

        X, y = verify_X_y(X, y)
        d = X.shape[1]
        self.n_features_in_ = X.shape[1]

        if self.intercepto:
            # não podemos penalizar o intercepto, então devemos cetralizar,
            # treinar, e depois encontrar o intercepto
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            Xprime = X - X_mean
            yprime = y - y_mean

        else:
            Xprime, yprime = X, y

        self.coef_ = np.linalg.solve(Xprime.T @ Xprime + self.lmbd * np.eye(d), Xprime.T @ yprime)
        self.intercept_ = float(y_mean - X_mean @ self.coef_) if self.intercepto else 0.0
        self._fitted = True
        return self
        
    def predict(self, X:np.ndarray):
        X = verify_array(X)

        if not self._fitted:
            raise NotFittedError(f'{self.__class__.__name__} was not fitted yet')

        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                f'Features dimension mismatch\nrequired: {self.n_features_in_}\ngiven: {X.shape[1]}'
            )

        return X @ self.coef_ + self.intercept_
