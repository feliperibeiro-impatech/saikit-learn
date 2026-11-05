import numpy as np

class MLPRegressor:
    def __init__(self, hidden_layers: tuple=(10,), learning_rate: float=0.01, n_epochs: int=1000, random_state: int = None):
        """_summary_

        Args:
            hidden_layers (tuple, opcional): Tamanhos das camadas ocultas. Padrão é (10,).
            learning_rate (float, optional): Passo do gradiente descende. Padrão é 0.01.
            n_epochs (int, optional): Vezes que o conjunto de treino é percorrido. Padrão é 1000.
            random_state (int, optional): Seed geradora dos pesos inciais. Padrão é None.
        """
        
        if not isinstance(hidden_layers, (tuple, list)):
            raise TypeError("hidden_layers deve ser uma tupla de inteiros.")
        
        if any((not isinstance(h, (int, np.integer))) or h <= 0 for h in hidden_layers):
            raise ValueError("Cada camada oculta deve ter um número inteiro positivo de neurônios.")
        
        if learning_rate <= 0:
            raise ValueError("learning_rate deve ser positivo.")
        
        if not isinstance(n_epochs, (int, np.integer)) or n_epochs <= 0:
            raise TypeError("n_epochs deve ser um inteiro positivo.")

        self._hidden = hidden_layers
        self._lr = learning_rate
        self._n_epochs = n_epochs
        self._rng = np.random.default_rng(random_state)
        self._weights, self._biases, self._n_features = None

    def fit(self, X: np.ndarray, y: np.ndarray):

        if not isinstance(X, np.ndarray):
            raise TypeError(f"X deve ser np.ndarray, recebeu {type(X).__name__}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y deve ser np.ndarray, recebeu {type(y).__name__}")
            
        if X.ndim != 2:
            raise ValueError("X deve ser 2D")
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y deve ser 1D e ter o mesmo número de amostras que X.")
        if X.shape[0] == 0:
            raise ValueError("X não pode ser vazio.")
        if not np.isfinite(X).all():
            raise ValueError("X contém NaN ou infinito.")
        if not np.isfinite(y).all():
            raise ValueError("y contém NaN ou infinito.")

        n_samples, self._n_features = X.shape
        y = y.reshape(-1, 1)

        size = [self._n_features, *self._hidden, 1]
    def predict(self, X, y):
        pass
    
    
    