import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.linspace(-3, 3, 50)
y = np.sin(X)

def lwr(x):
    w = np.exp(-(X-x)**2 / 0.18)
    A = np.c_[np.ones(len(X)), X]
    t = np.linalg.pinv(A.T @ (w[:,None]*A)) @ (A.T @ (w*y))
    return t[0] + t[1]*x

pred = [lwr(x) for x in X]

model = LinearRegression()
model.fit(X.reshape(-1,1), y)

plt.scatter(X, y)
plt.plot(X, pred, label="LWR")
plt.plot(X, model.predict(X.reshape(-1,1)), label="Linear")
plt.legend()
plt.show()