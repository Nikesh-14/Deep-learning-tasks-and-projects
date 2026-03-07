import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 5)
y_true = np.random.rand(100, 1)

X_with_bias = np.c_[np.ones((100, 1)), X]

w_with_bias = np.random.rand(6, 1)

mse_history = []
learning_rate = 0.01

for i in range(1000):
    predicted_y = np.dot(X_with_bias, w_with_bias)
    error = predicted_y - y_true
    w_gradient = (2 / len(X_with_bias)) * np.dot(X_with_bias.T, error)
    w_with_bias -= learning_rate * w_gradient
    mse = np.mean(error**2)
    mse_history.append(mse)


plt.plot(mse_history)
plt.title('MSE over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.grid()
plt.show()