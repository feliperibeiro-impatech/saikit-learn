import numpy as np
from typing import Optional

class DTNode:
  def __init__(
      self, 
      feature: Optional[int] = None,
      left: Optional["DTNode"] = None,
      right: Optional["DTNode"] = None,
      value: Optional[float] = None,
      gain: Optional[float] = None
      ):
    self.feature = feature
    self.left = left
    self.right = right
    self.value = value
    self.gain = gain

class DecisionTreeClassifier:
  def __init__(self, criterion: str = "gini"):
    self.criterion = criterion

  def fit(X: np.ndarray, y: np.ndarray):
    pass

  def _get_impurity(self, X):
    pass

  def predict(self, X: np.array) -> np.array:
    node = self.tree

    while node:
      pred_probabilities = node.pred_probability
      if X[node.feature] < node.feature_val:
        node = node.left
      else:
        node = node.right

    return pred_probabilities

  def _get_criterion_function(self):
    match self.criterion:
      case "gini":
        return 
      case "classification_error":
        return 
      case "entropy":
        return 

  def train(self, X: np.array, Y: np.array) -> None:
    pass

  def _p(k: float, rm: np.ndarray):
    return len(rm.filter(lambda x: x == k))

  def fit():
    pass
