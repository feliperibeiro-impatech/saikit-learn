from saikit_learn.utilidades.validacao import verify_X_y
import numpy as np

class OLS:
    '''Linear regression by ordinary least squares

    Parameters:
        intercepto: bool, default=True

    Attributes
    coef
    '''
    def __init__(self, intercepto:bool = True):
        self.intercepto = intercepto
        self._fitted = False

    def _design(self, x:np.ndarray):
        if self.intercepto:
            X = np.column_stack([np.ones(x.shape[0]), x])
            return X
        return x

    def fit(self, x:np.ndarray, y:np.ndarray):
        '''Train the model.
        Parameters:
            X : np.ndarray shape (n_sample, n_features)
            y = np.ndarray shape (n_sample)

        Returns:
            self : OLS
        
        '''
        X, y =  verify_X_y(x, y)
        X = self._design(X)

        beta = np.linalg.solve(X.T @ X, X.T @ y)

        self.intercept_ = beta[0] if self.intercepto else 0.0
        self.coef_ = self.beta[1:]
        self._fitted = True
        return self

    def predict(self, X:np.ndarray):
        '''Predict y for new sample.

        Parameters:
            X : np.ndarray shape (n_sample, n_features)
        
        Returns:
            ndarray shape (n_sample)

        Raise:

        '''
        if self._fitted:
            return X @ self.coef_ + self.intecept_

        else:
            raise ValueError(
                f'{self.__name__} was not fitted yet'
            )


