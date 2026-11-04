from saikit_learn.regressao.minimos_quadrados import OLS
import numpy as np
import pytest

# Initial setup
coef = np.array([2.0, -1.0, 0.5])
intercepto = 4.0

@pytest.fixture
def dados():
    '''Random X (50, 3) and y exact, whithout noise'''
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X @ coef + intercepto
    return X, y

@pytest.fixture
def dados_ruidosos():
    '''Random  X (1000, 3) and random y'''
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 3))
    y = X @ coef + intercepto + rng.normal(scale=0.1, size = 1000)
    return X, y



