import numpy as np
import time

X = np.random.rand(100, 3)
Y = np.random.rand(50, 3)

#using loops
now_1 = time.perf_counter()
D_loop = np.zeros((X.shape[0], Y.shape[0]))
for i in range(X.shape[0]):
    for j in range(Y.shape[0]):
        D_loop[i, j] = np.linalg.norm(X[i] - Y[j])
now_2 = time.perf_counter()
print(f"Distance matrix using loops:\n{D_loop}")
print(f"Time taken via loops: {now_2 - now_1:.8f} seconds")

#using broadcasting
now_1 = time.perf_counter()
D_broadcast = np.linalg.norm(X[:, np.newaxis] - Y, axis=2)
now_2 = time.perf_counter()
print(f"Time taken via broadcasting: {now_2 - now_1:.8f} seconds")
print(f"Distance matrix using broadcasting:\n{D_broadcast}")