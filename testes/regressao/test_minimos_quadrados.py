from saikit_learn.regressao.minimos_quadrados import OLS
from saikit_learn.utilidades.validacao import NotFittedError
import numpy as np
import pytest

# Initial setup
coef = np.array([2.0, -1.0, 0.5])
interception = 4.0

@pytest.fixture
def data():
    '''Random X (50, 3) and y exact, whithout noise'''
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X @ coef + interception
    return X, y

@pytest.fixture
def noise_data():
    '''Random  X (1000, 3) and random y'''
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 3))
    y = X @ coef + interception + rng.normal(scale=0.1, size = 1000)
    return X, y

# Testes numéricos
def test_recovers_coeficients_without_noise(data):
    X, y = data
    model = OLS().fit(X, y)
    np.testing.assert_allclose(model.coef_, coef, atol=1e-8)
    np.testing.assert_allclose(model.intercept_, interception, atol=1e-8)

def test_normal_equations(noise_data):
    X, y = noise_data
    A = np.column_stack([np.ones(X.shape[0]), X])
    beta_ref = np.linalg.solve(A.T @ A, A.T @ y)
    model = OLS().fit(X, y)
    np.testing.assert_allclose(model.intercept_, beta_ref[0], atol=1e-8)
    np.testing.assert_allclose(model.coef_, beta_ref[1:], atol=1e-8)

def test_orthogonal_residual(noise_data):
    X, y = noise_data
    model = OLS().fit(X, y)
    residual = y - model.predict(X)
    np.testing.assert_allclose(residual.sum(), 0.0, atol=1e-8)
    np.testing.assert_allclose(X.T @ residual, 0.0, atol=1e-8)

def test_correct_coeficients_with_noise(noise_data):
    X, y = noise_data
    model = OLS().fit(X, y)
    np.testing.assert_allclose(model.intercept_, interception, atol=.5)
    np.testing.assert_allclose(model.coef_, coef, atol=.5)

def test_predict_correctly(data):
    X, y = data
    model = OLS().fit(X, y)
    rng = np.random.default_rng(1)
    X_new = rng.normal(size=(100, 3))
    y_new = X_new @ coef + interception
    error = y_new - model.predict(X_new)
    np.testing.assert_allclose(error.sum(), 0.0, atol=1e-8)

def test_predict_correctly_with_noise(noise_data):
    X, y = noise_data
    model = OLS().fit(X, y)
    rng = np.random.default_rng(1)
    X_new = rng.normal(size=(1000, 3))
    y_new = X_new @ coef + interception + rng.normal(scale=0.1, size = 1000)
    error = y_new - model.predict(X_new)
    np.testing.assert_allclose(error.sum() / 1000, 0.0, atol=1e-2)

def test_without_interception():
    rng = np.random.default_rng(1)
    X = rng.normal(size = (50, 3))
    y = X @ coef
    model = OLS(intercepto=False).fit(X, y)
    np.testing.assert_allclose(model.coef_, coef, atol=1e-8)

def test_sum_constant_to_y(data):
    X, y = data
    base = OLS().fit(X, y)
    model = OLS().fit(X, y - 4.0)
    np.testing.assert_allclose(model.coef_, base.coef_, atol=1e-8)
    np.testing.assert_allclose(model.intercept_, base.intercept_ - 4.0, atol=1e-8)

def test_colinear_columns(data):
    X, y = data
    X_rd = np.column_stack([X, X[:, 0]])
    model = OLS().fit(X_rd, y)
    np.testing.assert_allclose(model.predict(X_rd), y, atol=1e-8)

def test_more_features_than_sample():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 10))
    y = rng.normal(size=5)
    model = OLS().fit(X, y)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-8)

# Testes de erros

def test_predict_before_fit(data):
    X, y = data
    with pytest.raises(NotFittedError):
        OLS().predict(X)

