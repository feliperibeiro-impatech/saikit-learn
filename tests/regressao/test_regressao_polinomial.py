from saikit_learn.regressao import PolReg
from saikit_learn.utilidades.validacao import NotFittedError
import numpy as np
import pytest

# Setup Inicial
def pol(x):
    return 1.2 + 3.4 * x - 7.34 * x ** 2 + 0.3 * x ** 3

@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = rng.uniform(-10, 10, size = (100, 1))
    return x, pol(x[:, 0])

@pytest.fixture
def noise_data():
    rng = np.random.default_rng(1)
    x = rng.uniform(-10, 10, size = (500, 1))
    y = pol(x[:, 0]) + rng.normal(scale = 0.5, size = 500)
    return x, y

# Testes Numéricos