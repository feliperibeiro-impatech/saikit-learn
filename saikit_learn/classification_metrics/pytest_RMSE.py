import numpy as np
import pytest 

from regression import RMSE

#ajustar depois, teste rápido para ver funcionamento
def test_MSE_5_large_numbers_case():
    """
    Test the MSE function with large numbers.
    The MSE should be 1e10
    """
    y_true = np.array([1e10, 2e10, 3e10])
    y_pred = np.array([1e10 + 80, 2e10 + 80, 3e10 + 80])
    assert RMSE(y_true, y_pred) == pytest.approx(80.0) 