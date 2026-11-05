import numpy as np

from typing import Optional

class TreeNode:
  def __init__(
      self, 
      feature: Optional[int] = None,
      left: Optional["TreeNode"] = None,
      right: Optional["TreeNode"] = None,
      value: Optional[float] = None,
      gain: Optional[float] = None,
      threshold: Optional[float] = None,
      n_samples: int = 0
      ):
    self.feature = feature
    self.left = left
    self.right = right
    self.value = value
    self.gain = gain
    self.threshold = threshold
    self.n_samples = n_samples

  @property
  def is_leaf(self) -> None:
    return self.feature is None


class DecisionTreeClassifier:
  def __init__(
      self, 
      criterion: str = "gini",
      min_samples_split: int = 2,
      min_samples_leaf: int = 1,
      min_gain: float = 0.0,
    ):
    self.criterion = criterion
    self.min_samples = 2

  def fit(X: np.ndarray, y: np.ndarray):
    pass

  def _get_best_split(self, X: np.ndarray, y:np.ndarray):
    """
       
    """
    n, n_features = X.shape
    n_classes = len(self.classes)
    impurity = self._get_criterion_function()
    onehot = np.eye(n_classes)[y]

    best = None
    best_score = np.inf

    for j in range(n_features):
      order = np.argsort(X[:, j], kind="stable")
      xs = X[order, j]

      cum = np.vstack([np.zeros((1, n_classes)), np.cumsum(onehot[order], axis=0)])

      thresholds = self._candidate_thresholds(xs)
      if thresholds.size == 0:
        continue

      n_left = np.searchsorted(xs, thresholds, side="right")
      n_right = n - n_left

      valid = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
      valid &= (n_left > 0) & (n_right > 0)

      thresholds, n_left, n_right = thresholds[valid], n_left[valid], n_right[valid]

      left_counts = cum[n_left]
      right_counts = cum[n] - left_counts
      q_left = impurity(left_counts / n_left[:, None])
      q_right = impurity(right_counts / n_right[:, None])
      scores = ((n_left * q_left) + (n_right * q_right)) / n

      i = int(np.argmin(scores))
      if scores[i] < best_score:
        best_score = float(scores[i])
        best = (j, float(thresholds[i]), best_score)

    return best






    


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

  def _p(k: int, y: np.ndarray):
    return float(np.mean(y == k))


  def _class_probabilites(self, y: np.ndarray) -> np.ndarray:
    n_classes = len(self._classes)

    return np.array([self._p(k, y) for k in range(n_classes)])

  def fit():
    pass
