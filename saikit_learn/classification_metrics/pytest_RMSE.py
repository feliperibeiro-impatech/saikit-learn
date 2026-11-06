import numpy as np
import pytest 

from regression import RMSE

# as this method is based on MSE function, we will just test a few cases to see if it is working properly

def test_RMSE_1_large_numbers_case():
    """
    Test the MSE function with large numbers.
    The MSE should be 80
    """
    y_true = np.array([1e10, 2e10, 3e10])
    y_pred = np.array([1e10 + 80, 2e10 + 80, 3e10 + 80])
    assert RMSE(y_true, y_pred) == pytest.approx(80.0) 

def test_RMSE_2_decimal_case():
    """
    Test the MSE function with a simple decimal case.
    The MSE should be 1/4
    """
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 1.5])
    assert RMSE(y_true, y_pred) == pytest.approx(0.5)

def test_RMSE_3_perfect_prediction():
    """
    Test the MSE function with perfect predictions.
    The MSE should be 0 when the predicted values are exactly equal to the true values
    """
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert RMSE(y_true, y_pred) == 0.0

def test_RMSE_4_small_numbers_case():
    """
    Test the MSE function with small numbers.
    The MSE should be 1e-11
    """
    y_true = np.array([1e-10, 2e-10, 3e-10])
    y_pred = y_true + 1e-11  
    assert RMSE(y_true, y_pred) == pytest.approx(1e-11, rel=1e-6, abs=0)
