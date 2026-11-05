from saikit_learn.utilidades.validacao import verify_X_y, NotFittedError
import numpy as np

class Ridge:
    def __init__(self, lmbd: float = 1.0, intercepto = True):
        self.lmbd = lmbd
        self.intercept_ = intercepto
        self._fitted = False

    def fit(self, X:np.ndarray, y:np.ndarray):
        if not np.isfinite(self.lmbd) or self.lmbd < 0:
            raise ValueError

        X, y = verify_X_y(X, y)

        if self.intercepto:
            # não podemos penalizar o intercepto, então devemos cetralizar,
            # treinar, e depois encontrar o intercepto
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            Xprime = X - X_mean
            yprime = y - y_mean

        else:
            Xprime, yprime = X, y

        d = X.shape[1]
        self.coef_ = np.linalg.solve(Xprime.T @ Xprime + self.lmbd * np.eye(d), Xprime.T @ y)
        self.intercept_ = float(y_mean - X_mean @ self.coef_) if self.intercept_ else 0.0
        self._fitted = True
        return self

        

    def predict(self, X:np.ndarray):
        pass
