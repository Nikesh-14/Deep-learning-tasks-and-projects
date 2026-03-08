import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def load_data():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  
    y = iris.target

    mask = y < 2
    X, y = X[mask], y[mask]
    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_params(dim):
    w = np.random.randn(dim) * 0.01
    b = np.random.randn() * 0.01
    return w, b

def train(X, y, lr, iterations):
    n_samples, n_features = X.shape
    w, b = initialize_params(n_features)
    loss_history = []

    for i in range(iterations):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        
        loss = -1/n_samples * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
        db = (1/n_samples) * np.sum(y_pred - y)
        
        w -= lr * dw
        b -= lr * db
        
        if i % 100 == 0:
            loss_history.append(loss)
            
    return w, b, loss, loss_history

def predict(X, w, b):
    probabilities = sigmoid(np.dot(X, w) + b)
    return [1 if i > 0.5 else 0 for i in probabilities]

def get_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

X, y = load_data()
learning_rates = [0.001, 0.01, 0.1, 1]


for lr in learning_rates:
    w, b, final_loss, loss_history = train(X, y, lr=lr, iterations=1000)
    preds = predict(X, w, b)
    acc = get_accuracy(y, preds)
    print(f"learning rate: {lr:<10}\tfinal loss: {final_loss:<12.6f}\naccuracy: {acc:<10.2f}%")
    print("-" * 52)
    plt.plot(loss_history, label=f'lr={lr}')

plt.title('Loss Convergence over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(True)
plt.show()