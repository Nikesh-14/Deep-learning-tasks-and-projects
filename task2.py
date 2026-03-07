import numpy as np
import time 

A = [[0, 0.22, 0.35],
     [0.15, 0.44, 0.32],
     [0.35, 0.21, 0.15],
     [0.25, 0.33, 0.28],
     [0.19, 0.11, 0.45]]
B = [0.33, 0.44, 0.23]

# Using loops
now_1 = time.perf_counter()

result_loop = []
for row in A:
    new_row = []
    for a_val, b_val in zip(row, B):
        new_row.append(a_val + b_val)
    result_loop.append(new_row)

now_2 = time.perf_counter()

print(f"Time taken: {now_2 - now_1:.8f} seconds")
print(f"Result using loops:\n {result_loop}")

#using numpy array
array_A = np.array(A)
array_B = np.array(B)
now_1 = time.perf_counter()
result_vectorized = array_A + array_B
now_2 = time.perf_counter()
print(f"Time taken: {now_2 - now_1:.8f} seconds")
print(f"Result using numpy array:\n {result_vectorized}")