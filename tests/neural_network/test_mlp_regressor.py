from saikit_learn.neural_network import MLPRegressor
import numpy as np
import pytest

# DADOS PARA TESTE E AUXILIARES

# Dados lineares sem ruído, Y = 2*x1 - 3x*2 + 1, com featurres em escala similar.
@pytest.fixture
def dados_linares():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(100,2))
    Y = 2 * X[:, 0] - 3 * X[:, 1] + 1
    return X, Y

# Poucos pontos com relação não linear, usados para testar overfit e queda da perda.
@pytest.fixture
def dados_pequenos():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(8, 1))
    Y = np.sin(X[:, 0])
    return X, Y

# _init_parameters

# Testa que os shapes de pesos e vieses seguem a lista sizes.
def test_init_parameters_shapes():
    modelo = MLPRegressor(random_state=0)
    modelo._init_parameters([3, 32, 16, 1])
    assert [W.shape for W in modelo._weights] == [(3, 32), (32, 16), (16, 1)]
    assert [b.shape for b in modelo._biases] == [(32,), (16,), (1,)]


# Testa que sem camada oculta existe um único W de shape (d, 1).
def test_init_parameters_sem_camada_oculta():
    modelo = MLPRegressor(hidden_layers=(), random_state=0)
    modelo._init_parameters([4, 1])
    assert len(modelo._weights) == 1
    assert modelo._weights[0].shape == (4, 1)


# Testa que os vieses começam em zero.
def test_init_parameters_vieses_zerados():
    modelo = MLPRegressor(random_state=0)
    modelo._init_parameters([3, 8, 1])
    assert all(np.all(b == 0) for b in modelo._biases)


# Testa que pesos não são todos iguais, o que quebraria a simetria entre neurônios.
def test_init_parameters_pesos_assimetricos():
    modelo = MLPRegressor(random_state=0)
    modelo._init_parameters([3, 8, 1])
    assert np.unique(modelo._weights[0]).size > 1


# Testa que o mesmo random_state gera os mesmos pesos e que sementes diferentes geram pesos diferentes.
def test_init_parameters_reprodutibilidade():
    a, b, c = (MLPRegressor(random_state=s) for s in (0, 0, 1))
    for m in (a, b, c):
        m._init_parameters([3, 8, 1])
    assert np.array_equal(a._weights[0], b._weights[0])
    assert not np.array_equal(a._weights[0], c._weights[0])


# Testa que a escala dos pesos segue He, com desvio próximo de sqrt(2 / n_in).
def test_init_parameters_escala_he():
    modelo = MLPRegressor(random_state=0)
    modelo._init_parameters([500, 500, 1])
    assert np.isclose(modelo._weights[0].std(), np.sqrt(2 / 500), rtol=0.05)


# _forward

# Testa a saída sem camada oculta, onde y = 2*x1 + 3*x2 + 1.
def test_forward_linear_conferido_a_mao():
    modelo = MLPRegressor(hidden_layers=())
    modelo._weights = [np.array([[2.0], [3.0]])]
    modelo._biases = [np.array([1.0])]
    activations, zs = modelo._forward(np.array([[1.0, 1.0], [0.0, 2.0]]))
    assert np.allclose(activations[-1], [[6.0], [7.0]])


# Testa que a ReLU zera valores negativos na camada oculta e que a saída não aplica ativação.
def test_forward_relu_na_oculta_e_identidade_na_saida():
    modelo = MLPRegressor(hidden_layers=(2,))
    modelo._weights = [np.array([[1.0, -1.0]]), np.array([[1.0], [1.0]])]
    modelo._biases = [np.zeros(2), np.array([-5.0])]
    activations, zs = modelo._forward(np.array([[2.0]]))
    assert np.allclose(activations[1], [[2.0, 0.0]])
    assert np.allclose(activations[-1], [[-3.0]])


# Testa que activations tem L+1 elementos (começando em X) e zs tem L.
def test_forward_tamanhos_das_listas():
    modelo = MLPRegressor(hidden_layers=(4, 3), random_state=0)
    modelo._init_parameters([2, 4, 3, 1])
    X = np.ones((5, 2))
    activations, zs = modelo._forward(X)
    assert len(activations) == 4 and len(zs) == 3
    assert np.array_equal(activations[0], X)
    assert [a.shape for a in activations] == [(5, 2), (5, 4), (5, 3), (5, 1)]

