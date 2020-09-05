from typing import Callable
from numpy import ndarray
from functions import add_10, square, cube, leaky_relu, sigmoid

import numpy as np
import matplotlib.pyplot as plt


def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001) -> ndarray:
    """
    Evaluates the derivative of a function "func" at every element in the input_ array
    """
    return (func(input_+delta) - func(input_-delta)) / (2*delta)


x: ndarray = np.arange(-10, 11)


y = add_10(x)
y_hat = deriv(add_10, x)
plt.plot(x, y, label="add_10")
plt.plot(x, y_hat, label="deriv of add_10")
plt.title("add_10 vs add_10'")
plt.legend()
plt.show()


y = square(x)
y_hat = deriv(square, x)
plt.plot(x, y, label="square")
plt.plot(x, y_hat, label="deriv of square")
plt.title("square vs square'")
plt.legend()
plt.show()


y = cube(x)
y_hat = deriv(cube, x)
plt.plot(x, y, label="cube")
plt.plot(x, y_hat, label="deriv of cube")
plt.title("cube vs cube'")
plt.legend()
plt.show()


y = leaky_relu(x)
y_hat = deriv(leaky_relu, x)
plt.plot(x, y, label="leaky_relu")
plt.plot(x, y_hat, label="deriv of leaky_relu")
plt.title("leaky_relu vs leaky_relu'")
plt.legend()
plt.show()


y = sigmoid(x)
y_hat = deriv(sigmoid, x)
plt.plot(x, y, label="sigmoid")
plt.plot(x, y_hat, label="deriv of sigmoid")
plt.title("sigmoid vs sigmoid'")
plt.legend()
plt.show()
