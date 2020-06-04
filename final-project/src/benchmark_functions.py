import numpy as np


def sinc(X):

    return np.sinc(X)


def hebbal(X):

    return -0.5 * (np.sin(40 * np.power(X - 0.85, 4)) * np.cos(2 * (X - 0.95)) + 0.5 * (X - 0.9) + 1)


def step(X):

    if X > 0:
        y = 1
    else:
        y = 0

    return y
