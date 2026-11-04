import numpy as np

class MQO:
    def __init__(self, intercepto:bool = True):
        self.intercepto = intercepto
        self.beta = None

    def _design(self, x:np.array):
        if self.intercepto:
            X = np.column_stack([np.ones(len(x)), x])
            return X
        return x

    def train(self, x:np.array, y:np.array):
        X = self._design(x)
        self.beta = np.linalg.solve(X.T @ X, X.T @ y)

    def predict(self, x:np.array):
        if self.beta is None:
            raise ValueError("Beta cannot be None")
        return self._design(x) @ self.beta


