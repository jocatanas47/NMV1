import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

N = 500

x11 = rnd.randn(N, 1)
x21 = rnd.randn(N, 1)

x12 = rnd.randn(N, 1) + 3
x22 = rnd.randn(N, 1) + 3

K1 = np.concatenate((x11, x21), axis=1)
K2 = np.concatenate((x12, x22), axis=1)
D1 = np.zeros((N, 1))
D2 = np.ones((N, 1))

plt.figure()
plt.plot(x11, x21, 'o')
plt.plot(x12, x22, 'x')
plt.show()
bias = np.ones((2*N, 1))

X = np.concatenate((K1, K2), axis=0)
X = np.concatenate((X, bias), axis=1)
D = np.concatenate((D1, D2), axis=0)

def predict(x, w):
    z = np.sum(x*w)
    if z >= 0:
        y = 1
    else:
        y = 0
    return y

def weight_update(d, y, x, lr):
    dw = lr*(d - y)*x
    return dw

W = rnd.randn(1, 3)*0.1
lr = 0.01
th = 0.002
for e in range(10):
    error = 0
    for i in range(2*N):
        x_tren = X[i, :]
        d_tren = D[i]
        y = predict(x_tren, W)
        dW = weight_update(d_tren, y, x_tren, lr)
        W += dW
        error += abs(d_tren - y)
    error /= np.shape(X)[0]
    print(error)
    if error < th:
        break

Ntest = 100
x1 = np.linspace(-2, 6, Ntest)
x2 = np.linspace(-2, 6, Ntest)
x1grid, x2grid = np.meshgrid(x1, x2)

x1grid = x1grid.reshape((1, Ntest**2))
x2grid = x2grid.reshape((1, Ntest**2))
bias = np.ones((1, Ntest**2))

grid = np.concatenate((x1grid, x2grid, bias), axis=0).T

Ypred = []
for g in grid:
    Ypred.append(predict(g, W))

K1test = grid[np.array(Ypred) == 0, :2]
K2test = grid[np.array(Ypred) == 1, :2]

plt.figure()
plt.plot(K1test[:, 0], K1test[:, 1], 'r.', alpha=0.1)
plt.plot(K2test[:, 0], K2test[:, 1], 'b.', alpha=0.1)
plt.plot(x11, x21, 'ro')
plt.plot(x12, x22, 'b*')
plt.show()