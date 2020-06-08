import numpy as np
import keras
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, RationalQuadratic, Matern, DotProduct

from active_learning import variance_based_active_learning, lola_active_learning, random_sequence_active_learning, \
    halton_sequence_active_learning, sobol_sequence_active_learning
from benchmark_functions import sinc, hebbal, step, problem15, problem20
from metrics import compute_ise_and_iv, compute_ise
from models import fit_gaussian_process, fit_neural_network
from plot_functions import plot_model, plot_ise_and_iv, plot_kernel_comparison, plot_activation_comparison
from sampling import ndgrid, initial_ccd


if __name__ == "__main__":

    # set the random seed
    np.random.seed(0)

    # for all f in a list
    f = sinc  # f = problem15  # f = sinc         # f = step  # f = hebbal  # f = sinc
    a = [-5]  # a = [-5]       # a = [-5]         # a = [-2]  # a = [0]     # a = [-5]
    b = [15]  # b = [5]        # b = [15]         # b = [2]   # b = [1]     # b = [5]
    fn_prefix = "sinc_shifted/"  # "problem15/"   # "sinc_shifted/"  # "step/"   # "hebbal/"   # "sinc/"

    n_random_trials = 5
    k_max = 50

    # method sweep
    surrogate_models        = ['gp', 'nn']
    active_learning_methods = [halton_sequence_active_learning, sobol_sequence_active_learning]  # [lola_active_learning, random_sequence_active_learning, variance_based_active_learning]
    active_learning_flags   = ['halton', 'sobol']  # ['lola', 'random', 'variance']

    # gp parameters
    kernels = [ConstantKernel(), RBF(), Matern(nu=1/2), Matern(nu=3/2), Matern(nu=5/2), RationalQuadratic(), DotProduct()]
    kernel_flags  = ['constant', 'rbf', 'matern12',   'matern32',   'matern52',   'rationalquad', 'dot']
    kernel_labels = ['Constant', 'RBF', 'Matérn-1/2', 'Matérn-3/2', 'Matérn-5/2', 'RQ',           'Dot Prod.']

    # nn parameters
    activations                   = ['relu', 'selu']
    activation_labels             = ['ReLU', 'SELU']
    hidden_units                  = [64]
    epochs                        = 10000
    stop_early_callback           = keras.callbacks.EarlyStopping(monitor='mse', min_delta=1e-6, patience=1500, verbose=0, restore_best_weights=True)
    reduce_learning_rate_callback = keras.callbacks.ReduceLROnPlateau(monitor='mse', factor=0.5, patience=500, min_lr=1e-5, verbose=0)
    callbacks                     = [stop_early_callback, reduce_learning_rate_callback]

    # query points
    X_pred = ndgrid(a, b, m=1001)

    # plot points
    X_plot = ndgrid(a, b, m=201)
    x_lims = (a[0], b[0])
    y_lims = (min(f(X_plot)), max(f(X_plot)))

    flags = ['', '']

    for fs, surrogate_model in enumerate(surrogate_models):                    # loop over surrogate model types

        flags[0] = surrogate_models[fs]

        for fa, active_learning_method in enumerate(active_learning_methods):  # loop over active learning methods

            flags[1] = active_learning_flags[fa]

            if surrogate_model == 'gp':                         # for gps

                ise_kernels = np.zeros((len(kernels), n_random_trials, k_max + 1))
                iv_kernels  = np.zeros((len(kernels), n_random_trials, k_max + 1))

                for fk, kernel in enumerate(kernels):                          # iterate over kernels

                    print(fk)

                    for n in range(n_random_trials):

                        print(n)

                        X_train_init = initial_ccd(a, b, m=3, add_random_sample=True)
                        y_train_init = f(X_train_init)

                        file_prefix = fn_prefix + flags[0] + '_' + flags[1] + "/" + kernel_flags[fk]
                        x_train_new = np.nan  # for null first plot
                        ise = []
                        iv = []

                        for k in range(k_max + 1):

                            if k == 0:
                                X_train = X_train_init
                                y_train = y_train_init

                            # re-fit model
                            gp = fit_gaussian_process(X_train, y_train, kernel)

                            # plot if desired
                            if k % k_max == 0:
                                filename = "plots/" + file_prefix + "/model_{}".format(n) + "_{}.svg".format(k)
                                plot_model(flags, gp, X_train, y_train, f, x_lims, y_lims, X_plot, x_new=x_train_new,
                                           filename=filename)

                            # compute ise, iv, and store values
                            ise_new, iv_new, y_pred, std_pred = compute_ise_and_iv(gp, f, a, b, X_pred)
                            ise = np.append(ise, ise_new)
                            iv  = np.append(iv, iv_new)
                            #print("Iteration %2.0i | ISE: %8.2e  | IV: %8.2e" % (k, ise[-1], iv[-1]))

                            # get new sample using active learning method
                            x_train_new, y_train_new = active_learning_method(gp, f, a, b, X_pred, std_pred, X_train)

                            # append new sample to training data
                            X_train = np.append(X_train, x_train_new.reshape(1, -1), axis=0)
                            y_train = np.append(y_train, y_train_new.reshape(1, -1), axis=0)

                            # sort training data (for LOLA & interval ordering)
                            sort_indices = np.argsort(X_train, axis=0)
                            X_train      = X_train[sort_indices].reshape(-1, 1)
                            y_train      = y_train[sort_indices].reshape(-1, 1)

                        filename = "plots/" + file_prefix + "/ise_iv_model_{}.svg".format(n)
                        plot_ise_and_iv(ise, iv, k_max, filename=filename)

                        ise_kernels[fk, n, :] = ise
                        iv_kernels[fk, n, :]  = iv

                # boxplot over all kernels (with specific learning method)
                plot_kernel_comparison(ise_kernels, iv_kernels, flags, kernel_labels, fn_prefix, n_random_trials, k_max)

            elif surrogate_model == 'nn' and flags[1] == 'variance':
                continue
            # for nns, iterate over activation functions
            elif surrogate_model == 'nn':

                ise_activation = np.zeros((len(activations), n_random_trials, k_max + 1))

                for fk, activation in enumerate(activations):

                    print(fk)

                    for n in range(n_random_trials):

                        print(n)

                        X_train_init = initial_ccd(a, b, m=3, add_random_sample=True)
                        y_train_init = f(X_train_init)

                        file_prefix = fn_prefix + flags[0] + '_' + flags[1] + "/" + activation
                        x_train_new = np.nan  # for null first plot
                        ise = []
                        iv  = []

                        for k in range(k_max + 1):

                            if k == 0:
                                X_train = X_train_init
                                y_train = y_train_init

                            # re-fit model
                            nn = fit_neural_network(X_train, y_train, hidden_units, activation, epochs, callbacks)

                            # plot if desired
                            if k % k_max == 0:
                                filename = "plots/" + file_prefix + "/{}".format(n) + "_{}.svg".format(k)
                                plot_model(flags, nn, X_train, y_train, f, x_lims, y_lims, X_plot, x_new=x_train_new,
                                           filename=filename)

                            # compute ise, iv, and store values
                            ise_new = compute_ise(nn, f, a, b, X_pred)
                            ise     = np.append(ise, ise_new)
                            print("Iteration %2.0i | ISE: %8.2e" % (k, ise[-1]))

                            # get new sample using active learning method
                            x_train_new, y_train_new = active_learning_method(nn, f, a, b, X_pred, [], X_train)

                            # append new sample to training data
                            X_train = np.append(X_train, x_train_new.reshape(1, -1), axis=0)
                            y_train = np.append(y_train, y_train_new.reshape(1, -1), axis=0)

                            # sort training data (for LOLA & interval ordering)
                            sort_indices = np.argsort(X_train, axis=0)
                            X_train = X_train[sort_indices].reshape(-1, 1)
                            y_train = y_train[sort_indices].reshape(-1, 1)

                        filename = "plots/" + file_prefix + "/ise_iv_model_{}.svg".format(n)
                        plot_ise_and_iv(ise, iv, k_max, filename=filename)

                        ise_activation[fk, n, :] = ise

                # boxplot over all activations (with specific learning method)
                plot_activation_comparison(ise_activation, flags, activation_labels, fn_prefix, n_random_trials, k_max)
