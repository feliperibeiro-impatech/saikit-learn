from collections import Counter
import pandas as pd
import numpy as np

class KNN:
    def __init__(self, k: int, label: str, data: pd.DataFrame) -> None:
        self._k = k
        self._label = label
        self._data = data

    def _get_distances_with_labels(self, point:pd.Series) -> pd.DataFrame:

        label = self._label
        data = self._data
        features = data[data.columns.drop(label)]

        distances = np.linalg.norm(features - point, axis=1)
        result = pd.DataFrame({"distances": distances, label : data[label]}).sort_values(by = "distances")

        return result

    def classify(self, input: pd.DataFrame) -> pd.DataFrame:
        
        label = self._label
        output = input.copy()
        predicted_labels = []

        for _, point in input.iterrows():
            distances_with_labels = self._get_distances_with_labels(point)

            k_neighbors = distances_with_labels[0 : self._k]
            majority = Counter(k_neighbors[label].to_numpy()).most_common(1)[0][0]  #Escolhe o primero rótulo por ser a dos pontos mais próximos
            predicted_labels.append(majority)

        output[label] = predicted_labels

        return output