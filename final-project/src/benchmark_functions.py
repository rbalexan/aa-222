import numpy as np


def sinc(X):

    return np.sinc(X)


def hebbal(X):

    return -0.5 * (np.sin(40 * np.power(X - 0.85, 4)) * np.cos(2 * (X - 0.95)) + 0.5 * (X - 0.9) + 1)


def step(X):

    return (X > 0)*1.0 - 0.5


# http://infinity77.net/global_optimization/test_functions_1d.html


def problem15(X):
    # on [-5, 5]
    return (np.power(X, 2) - 5*X + 6) / (np.power(X, 2) + 1)


def problem20(X):
    # on [-5, 5]
    return - (X - np.sin(X)) * np.exp(-np.power(X, 2))
