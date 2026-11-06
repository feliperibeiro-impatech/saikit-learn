from saikit_learn.regressao.ridge import Ridge
from saikit_learn.utilidades.validacao import NotFittedError
import numpy as np
import pytest

# Setup Inicial
coef = np.array([1.0, -2.0, 0.3])
interception = 4.5

@pytest.fixture
def data():
    '''X (50, 3) aleatório e y (50, 1) exato, sem ruído'''
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X @ coef + interception
    return X, y

@pytest.fixture
def noise_data():
    '''X (100, 3) aleatório e y (100, 1) com ruído'''
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X @ coef + interception + rng.normal(scale=0.1, size = 100)
    return X, y

