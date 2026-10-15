from collections import Counter
import pandas as pd
import numpy as np

class KNN:
    def __init__(self, k: int, label: str, data: pd.DataFrame, p: int = 2) -> None:
        self._k = k
        self._p = p
        self._label = label
        self._data = data
        if self._p not in [1,2]:
            raise ValueError

    def _get_distances_with_labels(self, point:pd.Series) -> pd.DataFrame:
        label = self._label
        data = self._data
        features = data[data.columns.drop(label)]

        point_features = point[features.column]

        diff = np.abs(features - point_features)
        distances = np.sum(diff ** self._p, axis=1) ** (1 / self._p)

        result = pd.DataFrame({"distances": distances, label: data[label]}).sort_values(by="distances")
        return result

    def classify(self, input: pd.DataFrame) -> pd.DataFrame:    
        label = self._label
        output = input.copy()
        predicted_labels = []

        for _, point in input.iterrows():
            distances_with_labels = self._get_distances_with_labels(point)

            k_neighbors = distances_with_labels[0 : self._k]

            #Escolhe o primero rótulo por ser a dos pontos mais próximos
            majority = Counter(k_neighbors[label].to_numpy()).most_common(1)[0][0]  
            predicted_labels.append(majority)

        output[label] = predicted_labels

        return output

    def score (self, test_data : pd.DataFrame) -> float:
        y_true = test_data[self._label].to_numpy()

        predictions = self.classify(test_data)
        y_pred = predictions[self._label].to_numpy()

        accuracy = np.mean(y_true == y_pred)
        return accuracy