import numpy as np
import chaospy


# noinspection PyUnusedLocal
def random_sequence_active_learning(model, f, a, b, X_pred, std_pred, X_train, k):

    # hack for 1D
    dims = len(a)
    x_train_new = np.random.random(dims) * (b[0] - a[0]) + a[0]
    y_train_new = f(x_train_new)

    return x_train_new, y_train_new


# noinspection PyUnusedLocal
def halton_sequence_active_learning(model, f, a, b, X_pred, std_pred, X_train, k):

    distribution = chaospy.J(chaospy.Uniform(a[0], b[0]))
    samples = distribution.sample(k+1, rule="halton")  # since k = 0 needs one sample

    x_train_new = samples[-1]
    y_train_new = f(x_train_new)

    return x_train_new, y_train_new


# noinspection PyUnusedLocal
def sobol_sequence_active_learning(model, f, a, b, X_pred, std_pred, X_train, k):

    distribution = chaospy.J(chaospy.Uniform(a[0], b[0]))
    samples = distribution.sample(k+2, rule="sobol")  # since k = 0 needs one sample that isn't 0.5

    x_train_new = samples[-1]
    y_train_new = f(x_train_new)

    return x_train_new, y_train_new


# noinspection PyUnusedLocal
def variance_based_active_learning(model, f, a, b, X_pred, std_pred, X_train, k):

    x_train_new = X_pred[np.argmax(std_pred)]
    y_train_new = f(x_train_new)

    return x_train_new, y_train_new


# noinspection PyUnusedLocal
def lola_active_learning(model, f, a, b, X_pred, std_pred, X_train, k):

    lola_error  = 0.0
    x_train_new = 0.0

    for i in range(X_train.shape[0] - 1):

        x_lower = X_train[i, :]
        x_upper = X_train[i + 1, :]
        x_mid = (x_lower + x_upper) / 2

        y_lower = f(x_lower)
        y_upper = f(x_upper)
        y_mid_pred = model.predict(x_mid.reshape(1, -1))
        y_mid_lola = (y_lower + y_upper) / 2

        lola_error_ = np.abs(y_mid_pred - y_mid_lola)
        # print(x_mid, y_mid_lola, y_mid_pred, lola_error_)

        if lola_error_ > lola_error:
            lola_error = lola_error_
            x_train_new = x_mid

    y_train_new = f(x_train_new)

    return x_train_new, y_train_new
