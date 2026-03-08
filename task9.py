import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X_raw = iris.data[:, [0, 3]]
y = iris.target
mask = y < 2 
X_raw, y = X_raw[mask], y[mask]

X_scaled = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X, y, lr=0.1, iterations=1000):
    n_samples, n_features = X.shape
    w = np.random.randn(n_features) * 0.01
    b = np.random.randn() * 0.01
    loss_history = []

    for i in range(iterations):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
        
        loss = -1/n_samples * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        loss_history.append(loss)

        dw = (1/n_samples) * np.dot(X.T, (y_hat - y))
        db = (1/n_samples) * np.sum(y_hat - y)
        
        w -= lr * dw
        b -= lr * db
            
    return loss_history, w, b

lr = 0.05
iters = 1000

hist_unscaled, w_u, b_u = train(X_raw, y, lr, iters)
hist_scaled, w_s, b_s = train(X_scaled, y, lr, iters)

def get_acc(X, y, w, b):
    preds = [1 if i > 0.5 else 0 for i in sigmoid(np.dot(X, w) + b)]
    return np.mean(y == preds) * 100

print(f"{'Unscaled':<12}\nFinal Loss: {hist_unscaled[-1]:<12.6f}\nAccuracy: {get_acc(X_raw, y, w_u, b_u):.2f}%")
print(f"{'Scaled':<12}\nFinal Loss: {hist_scaled[-1]:<12.6f}\nAccuracy: {get_acc(X_scaled, y, w_s, b_s):.2f}%")

plt.plot(hist_unscaled, label='Unscaled Features', color='red')
plt.plot(hist_scaled, label='Scaled Features', color='blue')
plt.title('Convergence: Unscaled vs Scaled Features')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()