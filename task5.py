import numpy as np

X = np.array([1000, 1001, 1002])
new_X = X - np.max(X) 

softmax = np.exp(new_X) / np.sum(np.exp(new_X))

print("Softmax probabilities:", softmax)