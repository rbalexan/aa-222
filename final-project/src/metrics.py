import numpy as np


def compute_ise_and_iv(gp, f, a, b, X_pred):

    # hack for 1D
    y_pred, std_pred = gp.predict(X_pred, return_std=True)
    ise              = np.sum((y_pred - f(X_pred)) ** 2) * (b[0] - a[0]) / len(y_pred)
    iv               = np.sum(std_pred ** 2)             * (b[0] - a[0]) / len(std_pred)

    return ise, iv, y_pred, std_pred


def compute_ise(nn, f, a, b, X_pred):

    y_pred = nn.predict(X_pred)
    ise = np.sum((y_pred - f(X_pred)) ** 2) * (b[0] - a[0]) / len(y_pred)

    return ise
