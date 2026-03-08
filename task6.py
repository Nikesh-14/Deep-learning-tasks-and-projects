import numpy as np

A = np.random.rand(5, 5)
determ = np.linalg.det(A)
print(f"Matrix A:\n{A}")
print(f"Determinant of A: {determ}")

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues of A: {eigenvalues}")
print(f"Eigenvectors of A:\n{eigenvectors}")

#verification of eigen equation A * v = λ * v
for i in range(len(eigenvalues)):
    left_side = np.dot(A, eigenvectors[:, i])
    right_side = eigenvalues[i] * eigenvectors[:, i]
    print(f"Verification for eigenvalue {eigenvalues[i]}: {np.allclose(left_side, right_side)}")

