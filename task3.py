import numpy as np

X = np.random.randint(0, 100, size=(10, 10))

print(f"Original array:\n{X}")
print(f"Array containing only greater than 50:\n{X[X > 50]}")

print(f"Array with even numbers replaced with -1:\n{np.where(X % 2 == 0, -1, X)}")

print(f"Indices of elements greater than 50:\n{np.where(X > 50)}")

print(f"mean of elements greater than 50: {np.mean(X[X > 50])}")