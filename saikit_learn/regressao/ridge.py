from saikit_learn.utilidades.validacao import verify_array, verify_X_y, NotFittedError
import numpy as np

class Ridge:
    def __init__(self, lmbd: float = 1.0, intercepto : bool = True):
        self.lmbd = lmbd
        self.intercepto = intercepto
        self._fitted = False

    def fit(self, X:np.ndarray, y:np.ndarray):
        try:
            lmbd = float(self.lmbd)

        except (TypeError, ValueError):
            raise ValueError(
                f'lambda deve ser um número; dado: {type(self.lmbd)}'
            )

        if not np.isfinite(lmbd) or lmbd < 0:
            raise ValueError(
                f'Lambda deve ser um número real não negativo; dado: {self.lmbd}'
            )

        X, y = verify_X_y(X, y)
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

        self.coef_ = np.linalg.solve(Xprime.T @ Xprime + self.lmbd * np.eye(self.n_features), Xprime.T @ yprime)
        self.intercept_ = float(y_mean - X_mean @ self.coef_) if self.intercepto else 0.0
        self._fitted = True
        return self
        
    def predict(self, X:np.ndarray):
        X = verify_array(X)

        if not self._fitted:
            raise NotFittedError(f'{self.__class__.__name__} não foi fittado ainda')

        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                f'A dimensão das features está incorreta\nnecessária: {self.n_features_in_}\ndado: {X.shape[1]}'
            )
        
        return X @ self.coef_ + self.intercept_
