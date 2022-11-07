import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def step(x):
    return np.piecewise(x, [x < 0.0, x > 0.0], [0, 1])


def relu(x):
    # return np.piecewise(x, [x < 0, x > 0], [0, lambda x: x])
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


if __name__ == '__main__':
    # Generates 10 values from 10 to -10
    x = np.linspace(10, -10, 100)

    # Using function sigmoid
    plt.plot(x, sigmoid(x))
    plt.show()

    # Using function step
    plt.plot(x, step(x))
    plt.show()

    # Using function relu
    plt.plot(x, relu(x))
    plt.show()

    # Using function tanh
    plt.plot(x, tanh(x))
    plt.show()
