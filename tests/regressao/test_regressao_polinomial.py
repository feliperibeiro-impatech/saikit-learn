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
def test_recovers_exact_poly_coef(data):
    X, y = data
    model = PolReg(d=3).fit(X, y)
    X_new = np.linspace(-10, 10, 100)[:, None]
    np.testing.assert_allclose(model.predict(X_new), pol(X_new[:, 0]), atol = 1e-8)


def test_recovers_noise_poly_coef(noise_data):
    X, y = noise_data
    model = PolReg(d=3).fit(X, y)
    X_new = np.linspace(-10, 10, 100)[:, None]
    np.testing.assert_allclose(model.predict(X_new), pol(X_new[:, 0]), atol = 0.5)


