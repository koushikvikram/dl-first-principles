from typing import List, Callable
from numpy import ndarray
from functions import square, sigmoid


import numpy as np
import matplotlib.pyplot as plt


# a function that takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[ndarray], ndarray]

# a chain is a list of functions
Chain = List[Array_Function]


# define how data goes through a chain of length 2
def chain_length_2(chain: Chain, input_: ndarray) -> ndarray:
    """
    Evaluates 2 functions in a row, in a 'Chain'
    """
    assert len(chain) == 2, "Length of input 'chain' should be 2"

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(input_))


x: ndarray = np.arange(-10, 11)


y: ndarray = chain_length_2([square, sigmoid], x)
plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title("sigmoid(square(x))")


y: ndarray = chain_length_2([sigmoid, square], x)
plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.title("square(sigmoid(x))")


plt.tight_layout()
plt.show()
