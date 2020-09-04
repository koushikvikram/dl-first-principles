import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray


def square(input_: ndarray) -> ndarray:
    """
    Square each element of input_ array
    """
    return np.power(input_, 2)


def leaky_relu(input_: ndarray) -> ndarray:
    """
    Apply leaky relu on each element of input_ array
    """
    return np.maximum(0.2*input_, input_)


x: ndarray = np.arange(-10, 11)
y: ndarray = square(x)

plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title("square")


y: np.ndarray = leaky_relu(x)

plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.title("leaky relu")
plt.show()
