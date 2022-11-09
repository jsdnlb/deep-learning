import numpy as np


def mean_squared_error(x, x_hat, derivate=False):
    if derivate:
        return (x_hat - x)
    return np.mean((x_hat - x) ** 2)


if __name__ == '__main__':
    real = np.array([0, 0, 1, 1])
    prediction = np.array([0.9, 0.1, 1, 0.5])
    
    # Using loss function 
    print(mean_squared_error(real, prediction))
    
    # Using loss function with derivate
    print(mean_squared_error(real, prediction, True))
