from typing import Callable
from numpy import ndarray
from functions import square

import numpy as np
import matplotlib.pyplot as plt


def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001) -> ndarray:
    """
    Evaluates the derivative of a function "func" at every element in the input_ array
    """
    return (func(input_+delta) - func(input_-delta)) / (2*delta)


# x: ndarray = np.arange(-10, 11)
# y = square(x)
# y_hat = deriv(square, x)

x: ndarray = np.arange(-10, 11)
y = add_10(x)
y_hat = deriv(add_10, x)


plt.plot(x, y)
plt.plot(x, y_hat)
plt.show()
