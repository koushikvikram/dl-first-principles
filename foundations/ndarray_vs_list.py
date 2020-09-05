from numpy import ndarray
from typing import List

import numpy as np


# operations on list
print("Python list operations:")
a: List = [1, 2, 3]
b: List = [4, 5, 6]
print("a+b:", a+b)
try:
    print(a*b)
except TypeError:
    print("a*b has no meaning for Python lists")
print()


# operations on ndarray
print("Numpy Array operations:")
a: ndarray = np.array([1, 2, 3])
b: ndarray = np.array([4, 5, 6])
print("a+b:", a+b)
print("a*b:", a*b)


# operations along axis of ndarray
a: ndarray = np.array([[1, 2, 3],
              [4, 5, 6]])
print("a:", a)
print("a.sum(axis=0):", a.sum(axis=0))
print("a.sum(axis=1):", a.sum(axis=1))


# broadcasting ndarray
a: ndarray = np.array([[1, 2, 3],
                       [4, 5, 6]])
b: ndarray = np.array([10, 20, 30])
print("a+b:\n", a+b)
