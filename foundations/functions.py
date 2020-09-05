import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray


def add_10(input_: ndarray) -> ndarray:
    return input_+10


def square(input_: ndarray) -> ndarray:
    """
    Square each element of input_ array
    """
    return np.power(input_, 2)


def cube(input_: ndarray) -> ndarray:
    return np.power(input_, 3)


def leaky_relu(input_: ndarray) -> ndarray:
    """
    Apply leaky relu on each element of input_ array
    """
    return np.maximum(0.2*input_, input_)


def sigmoid(input_: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-input_))


x: ndarray = np.arange(-10, 11)


y: ndarray = add_10(x)
plt.subplot(1, 5, 1)
plt.plot(x, y)
plt.title("add_10")


y: ndarray = square(x)
plt.subplot(1, 5, 2)
plt.plot(x, y)
plt.title("square")


y: ndarray = cube(x)
plt.subplot(1, 5, 3)
plt.plot(x, y)
plt.title("cube")


y: ndarray = leaky_relu(x)
plt.subplot(1, 5, 4)
plt.plot(x, y)
plt.title("leaky relu")


y: ndarray = sigmoid(x)
plt.subplot(1, 5, 5)
plt.plot(x, y)
plt.title("sigmoid")


plt.tight_layout()
plt.show()
