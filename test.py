import numpy as np

print(np.random.permutation(6))

a = np.array([1, 2, 3, 4, 5, 6])
b = np.random.permutation(a)
index = b[0: 3]
print(index)
print(a[index])
