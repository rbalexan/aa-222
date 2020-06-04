import numpy as np
from itertools import product


def initial_ccd(a, b, m, add_random_sample=False):

    X = ndgrid(a, b, m)

    if add_random_sample:
        # hack for 1D
        dims = len(a)
        x_random = np.random.random(dims).reshape(1, -1) * (b[0] - a[0]) + a[0]
        X = np.append(X, x_random, axis=0)

    return X


def ndgrid(a, b, m):
    
    dims = len(a)
    AB = []

    for dim in range(dims):
        AB.append(np.linspace(a[dim], b[dim], num=m))

    return np.array([x for x in product(*AB)])
