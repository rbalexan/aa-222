import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, RationalQuadratic
from itertools import product
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU

def initial_ccd(a, b, m=3):

    dims = len(a)
    AB = []

    for dim in range(dims):
        AB.append(np.linspace(a[dim], b[dim], num=m))

    return np.array([x for x in product(*AB)])


def ndgrid(a, b, m=10):

    return initial_ccd(a, b, m=m)


def plot_gp_variance(gp, X_train, y_train, f, x_lims, y_lims, X, x_new=np.nan, n_std=2, save=True, filename="tmp.svg", padding=0.1):

    fig = plt.figure()

    # plot GP
    y_pred_gp, std_pred = gp.predict(X, return_std=True)
    plt.fill_between(np.squeeze(X), np.squeeze(y_pred_gp)+n_std*std_pred, np.squeeze(y_pred_gp)-n_std*std_pred,
                     color=(52/255, 122/255, 235/255), alpha=0.6, lw=0, label="GP predicted 2$\sigma$")
    plt.plot(X, y_pred_gp, c=(0.2, 0.2, 0.2), label="GP predicted mean")

    plot(fig, X_train, y_train, f, x_lims, y_lims, X, x_new, save, filename, padding)


def plot_gp_lola(gp, X_train, y_train, f, x_lims, y_lims, X, x_new=np.nan, n_std=2, save=True, filename="tmp.svg", padding=0.1):

    fig = plt.figure()

    # plot GP
    y_pred_gp, std_pred = gp.predict(X, return_std=True)
    plt.fill_between(np.squeeze(X), np.squeeze(y_pred_gp)+n_std*std_pred, np.squeeze(y_pred_gp)-n_std*std_pred,
                     color=(52/255, 122/255, 235/255), alpha=0.6, lw=0, label="GP predicted 2$\sigma$")
    plt.plot(X, y_pred_gp, c=(0.2, 0.2, 0.2), label="GP predicted mean")

    # plot local linear approximation
    plt.plot(X_train, y_train, marker="o", color=(0.4, 0.4, 0.4), label="local linear approx.")

    plot(fig, X_train, y_train, f, x_lims, y_lims, X, x_new, save, filename, padding)


def plot_nn_lola(nn, X_train, y_train, f, x_lims, y_lims, X, x_new=np.nan, save=True, filename="tmp.svg", padding=0.1):

    fig = plt.figure()

    # plot NN
    plt.plot(X, nn.predict(X), label="NN predicted mean")

    # plot local linear approximation
    plt.plot(X_train, y_train, marker="o", color=(0.4, 0.4, 0.4), label="local linear approx.")

    plot(fig, X_train, y_train, f, x_lims, y_lims, X, x_new, save, filename, padding)


def plot(fig, X_train, y_train, f, x_lims, y_lims, X, x_new, save, filename, padding):

    # plot true function
    plt.plot(X, f(X), c="black", label="true function")

    # plot training examples
    plt.scatter(X_train, y_train)

    # plot new training example
    if x_new != np.nan:
        plt.plot([x_new, x_new], [f(x_new), f(x_new)], c="magenta", marker="o", label="new sample")

    # format plot
    x_w = x_lims[1] - x_lims[0]
    plt.xlim((x_lims[0] - padding * x_w, x_lims[1] + padding * x_w))

    y_w = y_lims[1] - y_lims[0]
    plt.ylim((y_lims[0] - padding * y_w, y_lims[1] + padding * y_w))

    plt.legend()

    if save:
        plt.savefig(filename, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def initialize_gp(kernel):

    gp = GaussianProcessRegressor(kernel=kernel)

    return gp


def plot_ise_and_iv(ise, iv, k_max, save=True, filename="tmp.svg"):

    fig = plt.figure()
    plt.plot(range(k_max + 1), ise, marker="o", label="Integrated Squared Error")

    if not all(np.isnan(iv)):
        plt.plot(range(k_max + 1), iv,  marker="x", label="Integrated Variance")
    plt.yscale("log")
    plt.xlim((0, k_max))
    plt.xlabel("Iteration")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(filename, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def sinc(X):

    return np.sinc(X)


def hebbal(X):

    return -0.5 * (np.sin(40 * np.power(X - 0.85, 4)) * np.cos(2 * (X - 0.95)) + 0.5 * (X - 0.9) + 1)


def lola_active_learning_nn(f, a, b, hidden_units_list, epochs, k_max, file_prefix):

    X_train = initial_ccd(a, b)
    y_train = f(X_train)

    # query points
    X_pred = ndgrid(a, b, m=201)

    # plot points
    X_plot = ndgrid(a, b, m=201)

    nn = initialize_neural_network(hidden_units_list)

    x_train_new = np.nan
    x_lims = (a[0], b[0])
    y_lims = (min(f(X_plot)), max(f(X_plot)))
    ise = []

    for k in range(k_max + 1):

        # update and retrain model
        if k > 0:
            X_train = np.append(X_train, x_train_new.reshape(1, -1), axis=0)
            y_train = np.append(y_train, y_train_new.reshape(1, -1), axis=0)

            sort_idxs = np.argsort(X_train, axis=0)

            X_train = X_train[sort_idxs].reshape(-1, 1)
            y_train = y_train[sort_idxs].reshape(-1, 1)

        nn.fit(X_train, y_train, epochs=epochs, verbose=0)

        filename = "plots/" + file_prefix + "/{}.svg".format(k)
        plot_nn_lola(nn, X_train, y_train, f, x_lims, y_lims, X_plot, x_new=x_train_new, filename=filename)

        yq_pred = nn.predict(X_pred)
        ise = np.append(ise, np.sum((yq_pred - f(X_pred)) ** 2) * (b[0] - a[0]) / len(yq_pred))
        print("Iteration %2.0i | ISE: %8.2e" % (k, ise[-1]))

        # get our new point for the nn
        x_train_new, y_train_new = lola_active_learning(nn, f, X_train)

    filename = "plots/" + file_prefix + "/ise_iv_{}.svg".format(k_max)
    plot_ise_and_iv(ise, [np.nan], k_max, filename=filename)

    return ise, np.nan


def lola_active_learning(model, f, X_train):

    lola_error = 0.0

    for i in range(X_train.shape[0] - 1):

        x_lower = X_train[i, :]
        x_upper = X_train[i + 1, :]
        x_mid   = (x_lower + x_upper) / 2

        y_lower    = f(x_lower)
        y_upper    = f(x_upper)
        y_mid_pred = model.predict(x_mid)
        y_mid_lola = (y_lower + y_upper) / 2

        lola_error_ = np.abs(y_mid_pred - y_mid_lola)
        #print(x_mid, y_mid_lola, y_mid_pred, lola_error_)

        if lola_error_ > lola_error:
            lola_error  = lola_error_
            x_train_new = x_mid

    y_train_new = f(x_train_new)

    return x_train_new, y_train_new


def variance_based_active_learning(f, X_pred, std_pred):

    x_train_new = X_pred[np.argmax(std_pred)]
    y_train_new = f(x_train_new)

    return x_train_new, y_train_new


def variance_based_active_learning_gp(f, a, b, kernel, k_max, file_prefix):

    X_train = initial_ccd(a, b)
    y_train = f(X_train)

    # query points
    X_pred = ndgrid(a, b, m=201)

    # plot points
    X_plot = ndgrid(a, b, m=201)

    gp = initialize_gp(kernel)

    x_train_new = np.nan
    x_lims = (a[0], b[0])
    y_lims = (min(f(X_plot)), max(f(X_plot)))
    ise = []
    iv = []

    for k in range(k_max+1):

        # update and retrain model
        if k > 0:
            X_train = np.append(X_train, x_train_new.reshape(1, -1), axis=0)
            y_train = np.append(y_train, y_train_new.reshape(1, -1), axis=0)

        gp = gp.fit(X_train, y_train)

        filename = "plots/" + file_prefix + "/{}.svg".format(k)
        plot_gp_variance(gp, X_train, y_train, f, x_lims, y_lims, X_plot, x_new=x_train_new, filename=filename)

        yq_pred, std_pred = gp.predict(X_pred, return_std=True)
        ise = np.append(ise, np.sum((yq_pred - f(X_pred))**2) * (b[0] - a[0]) / len(yq_pred))
        iv  = np.append(iv,  np.sum(std_pred ** 2) * (b[0] - a[0]) / len(std_pred))
        print("Iteration %2.0i | ISE: %8.2e  | IV: %8.2e" % (k, ise[-1], iv[-1]))

        # get our new point for the gp
        x_train_new, y_train_new = variance_based_active_learning(f, X_pred, std_pred)

    filename = "plots/" + file_prefix + "/ise_iv_{}.svg".format(k_max)
    plot_ise_and_iv(ise, iv, k_max, filename=filename)

    return ise, iv


def lola_active_learning_gp(f, a, b, kernel, k_max, file_prefix):

    X_train = initial_ccd(a, b)
    y_train = f(X_train)

    # query points
    X_pred = ndgrid(a, b, m=201)

    # plot points
    X_plot = ndgrid(a, b, m=201)

    gp = initialize_gp(kernel)

    x_train_new = np.nan
    x_lims = (a[0], b[0])
    y_lims = (min(f(X_plot)), max(f(X_plot)))
    ise = []
    iv = []

    for k in range(k_max+1):

        # update and retrain model
        # update and retrain model
        if k > 0:
            X_train = np.append(X_train, x_train_new.reshape(1, -1), axis=0)
            y_train = np.append(y_train, y_train_new.reshape(1, -1), axis=0)

            sort_idxs = np.argsort(X_train, axis=0)

            X_train = X_train[sort_idxs].reshape(-1, 1)
            y_train = y_train[sort_idxs].reshape(-1, 1)

        gp = gp.fit(X_train, y_train)

        filename = "plots/" + file_prefix + "/{}.svg".format(k)
        plot_gp_lola(gp, X_train, y_train, f, x_lims, y_lims, X_plot, x_new=x_train_new, filename=filename)

        yq_pred, stdq_pred = gp.predict(X_pred, return_std=True)
        ise = np.append(ise, np.sum((yq_pred - f(X_pred))**2) * (b[0] - a[0]) / len(yq_pred))
        iv  = np.append(iv,  np.sum(stdq_pred ** 2) * (b[0] - a[0]) / len(stdq_pred))
        print("Iteration %2.0i | ISE: %8.2e  | IV: %8.2e" % (k, ise[-1], iv[-1]))

        # get our new point for the gp
        x_train_new = X_pred[np.argmax(stdq_pred)]
        y_train_new = f(x_train_new)

    filename = "plots/" + file_prefix + "/ise_iv_{}.svg".format(k_max)
    plot_ise_and_iv(ise, iv, k_max, filename=filename)

    return ise, iv


def initialize_neural_network(hidden_units_list):

    model = Sequential()

    for i, hidden_units in enumerate(hidden_units_list):

        if i == 0:
            model.add(Dense(hidden_units, input_dim=1, activation='selu'))
            #model.add(LeakyReLU())
        else:
            model.add(Dense(hidden_units, activation='selu'))

    model.add(Dense(1, activation=None))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    return model


if __name__ == "__main__":

    a = [-5]
    b = [5]

    k_max = 30
    file_prefix = "sinc"
    kernel = RBF()
    hidden_units = [32]
    epochs = 5000

    #gp_var_ise,  gp_var_iv  = variance_based_active_learning_gp(sinc, a, b, kernel, k_max, file_prefix + "/gp_variance")
    #gp_lola_ise, gp_lola_iv = lola_active_learning_gp(sinc, a, b, kernel, k_max, file_prefix + "/gp_lola")
    nn_lola_ise, nn_lola_iv = lola_active_learning_nn(sinc, a, b, hidden_units, epochs, k_max, file_prefix + "/nn_lola")

    a = [0]
    b = [1]

    k_max = 30
    file_prefix = "hebbal"
    kernel = RationalQuadratic()
    hidden_units = [32]

    #gp_var_ise,  gp_var_iv  = variance_based_active_learning_gp(hebbal, a, b, kernel, k_max, file_prefix + "/gp_variance")
    #gp_lola_ise, gp_lola_iv = lola_active_learning_gp(hebbal, a, b, kernel, k_max, file_prefix + "/gp_lola")
    #nn_lola_ise, nn_lola_iv = lola_active_learning_nn(hebbal, a, b, hidden_units, epochs, k_max, file_prefix + "/nn_lola")
