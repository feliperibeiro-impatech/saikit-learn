import numpy as np

class KNN:
    def __init__(self, k: int, p: int = 2) -> None:
        if self._p not in [1,2]:
            raise ValueError("p deve ser 1 ou 2.")
        self._k = k
        self._p = p
        self._X = None
        self._Y = None

    def fit(self, X, Y) -> None:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        
        if len(X) != len(Y):
            raise ValueError("X e Y devem ser de mesmo tamanho") 

        return self

    def _get_distances_with_labels(self, x: np.array) -> np.array:
        
        diff = np.abs(self._X - x)
        distances = np.sum(diff ** self._p, axis=1) ** (1 / self._p)

        result = np.argsort(distances, kind='stable')
        
        return self._Y[result]

    def predict(self, X: np.array) -> np.array:
        X = np.asarray(X, dtype=np.float64)
        
        predicted_labels = []

        for x in X:
            sorted_labels = self._get_distances_with_labels(x)

            k_neighbors = sorted_labels[:self._k]

            #Escolhe o primero rótulo por ser a dos pontos mais próximos
            label, first_pos, count = np.unique(k_neighbors, return_index=True, return_counts=True)
            candidates = count == count.max()
            predicted_labels.append(label[np.argmin(first_pos[candidates])])

        return np.array(predicted_labels)
