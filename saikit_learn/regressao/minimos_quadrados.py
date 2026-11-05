from saikit_learn.utilidades.validacao import verify_array, verify_X_y, NotFittedError
import numpy as np

class OLS:    
    def __init__(self, intercepto:bool = True):
        self.intercepto = intercepto
        self._fitted = False

    def _design(self, x:np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            x (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """        
        if self.intercepto:
            X = np.column_stack([np.ones(x.shape[0]), x])
            return X
        return x

    def fit(self, x:np.ndarray, y:np.ndarray):
        """_summary_

        Args:
            x (np.ndarray): _description_
            y (np.ndarray): _description_

        Returns:
            _type_: _description_
        """           
        X, y =  verify_X_y(x, y)
        self.n_features_in_ = X.shape[1]
        X = self._design(X)

        beta = np.linalg.solve(X.T @ X, X.T @ y)

        self.intercept_ = float(beta[0]) if self.intercepto else 0.0
        self.coef_ = beta[1:] if self.intercepto else beta
        self._fitted = True
        return self

    def predict(self, X:np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            X (np.ndarray): _description_

        Raises:
            ValueError: _description_
            NotFittedError: _description_

        Returns:
            np.ndarray: _description_
        """        '''Predict y for new sample.

        Parameters:
            X : np.ndarray shape (n_sample, n_features)
        
        Returns:
            ndarray shape (n_sample)

        Raise:
            NotFittedError

        '''
        X = verify_array(X)

        if not self._fitted:
            raise NotFittedError(
                f'{self.__class__.__name__} was not fitted yet'
                )
        
        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                f'Features dimension mismatch\nrequired: {self.n_features_in_}\ngiven: {X.shape[1]}'
                             )
        return X @ self.coef_ + self.intercept_

