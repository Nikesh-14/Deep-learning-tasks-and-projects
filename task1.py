import numpy as np
from datetime import datetime

# normal approach
x = np.random.rand(1_000_000)

now_1 = datetime.now()
f_x = 0
for values in x:
    f_x += values ** 2 + 2 * values + 1
now_2 = datetime.now()
print(f"Time taken: {now_2 - now_1}")
print(f"f(x) = {f_x}: normal approach")

# vectorized approach
now_1 = datetime.now()
f_x_vectorized = np.sum(x ** 2 + 2 * x + 1)
now_2 = datetime.now()
print(f"Time taken: {now_2 - now_1}")
print(f"f(x) = {f_x_vectorized}: vectorized approach")

