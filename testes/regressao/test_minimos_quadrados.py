from minimos_quadrados import MQO
import numpy as np
import matplotlib.pyplot as plt

n, p = 100, 1
x = np.random.random((n, p))

def f(t):
    return - np.pi * t.ravel() + 1/2

eps = np.random.normal(scale = 0.5, size = n)
y = f(x) + eps

# treino
model = MQO()
model.train(x, y)
print(model.beta)

x_n = np.linspace(0, 1, 200).reshape(-1, 1)
f_hat = model.predict(x_n)

plt.scatter(x, y, alpha = 0.6, label = 'data')
plt.plot(x_n, f_hat, label = 'prediction', color = 'red', linestyle = '-')
plt.plot(x_n, f(x_n), label = 'real function', color = 'black', linestyle = '--')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()


