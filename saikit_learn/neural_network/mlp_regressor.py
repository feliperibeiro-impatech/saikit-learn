import numpy as np

class MLPRegressor:
    def __init__(self, hidden_layer_sizes=(10,), learning_rate=0.01, n_epochs=1000, random_state=None):

        if not isinstance(hidden_layer_sizes, (tuple, list)):
            raise ValueError("hidden_layer_sizes deve ser uma tupla de inteiros.")
        
        if any((not isinstance(h, (int, np.integer))) or h <= 0 for h in hidden_layer_sizes):
            raise ValueError("Cada camada oculta deve ter um número inteiro positivo de neurônios.")
        
        if learning_rate <= 0:
            raise ValueError("learning_rate deve ser positivo.")
        
        if not isinstance(n_epochs, (int, np.integer)) or n_epochs <= 0:
            raise ValueError("n_epochs deve ser um inteiro positivo.")

        self._hidden = hidden_layer_sizes
        self._lr = learning_rate
        self._n_epochs = n_epochs
        self._rng = np.random.default_rng(random_state)
        self._weights, self._biases, self._n_features = None

    def fit(self, X, Y):
        pass
    
    def predict(self, X, Y):
        pass
    
    
    