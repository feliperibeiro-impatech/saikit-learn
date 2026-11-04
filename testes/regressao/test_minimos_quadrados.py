from saikit_learn.regressao.minimos_quadrados import OLS
import numpy as np
import pytest

# Initial setup
coef = np.array([2.0, -1.0, 0.5])
intercepto = 4.0

@pytest.fixture
def data():
    '''Random X (50, 3) and y exact, whithout noise'''
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X @ coef + intercepto
    return X, y

@pytest.fixture
def noise_data():
    '''Random  X (1000, 3) and random y'''
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 3))
    y = X @ coef + intercepto + rng.normal(scale=0.1, size = 1000)
    return X, y

# Testes numéricos
def test_recovers_coeficients_whithout_noise(data):
    X, y = data
    model = OLS().fit(X, y)
    np.testing.assert_allclose(model.coef_, coef, atol=1e-8)
    np.testing.assert_allclose(model.intercept_, intercepto, atol=1e-8)



