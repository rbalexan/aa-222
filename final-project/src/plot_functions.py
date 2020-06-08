import numpy as np
import matplotlib.pyplot as plt


def plot_model(flags, model, X_train, y_train, f, x_lims, y_lims, X, x_new=np.nan, n_std=2, save=True, filename="tmp.svg", padding=0.1):

    fig = plt.figure()

    # plot model
    if flags[0] == 'gp':

        y_pred_gp, std_pred = model.predict(X, return_std=True)
        plt.fill_between(np.squeeze(X), np.squeeze(y_pred_gp)+n_std*std_pred, np.squeeze(y_pred_gp)-n_std*std_pred,
                         color=(52/255, 122/255, 235/255), alpha=0.6, lw=0, label="GP predicted 2$\sigma$")
        plt.plot(X, y_pred_gp, c=(0.2, 0.2, 0.2), label="GP predicted mean")

    elif flags[0] == 'nn':

        plt.plot(X, model.predict(X), label="NN predicted mean")

    # plot LOLA if it is the method
    if flags[1] == 'lola':

        plt.plot(X_train, y_train, marker="o", color=(0.4, 0.4, 0.4), label="local linear approx.")

    plot_general(fig, X_train, y_train, f, x_lims, y_lims, X, x_new, save, filename, padding)


def plot_general(fig, X_train, y_train, f, x_lims, y_lims, X, x_new, save, filename, padding):

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
    plt.xlabel("x")

    y_w = y_lims[1] - y_lims[0]
    plt.ylim((y_lims[0] - padding * y_w, y_lims[1] + padding * y_w))
    plt.ylabel("y")

    plt.grid(which='both')
    plt.legend()

    if save:
        plt.savefig(filename, dpi=300)
        plt.close(fig)
    else:
        plt.show()


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


def plot_kernel_comparison(ise_kernels, iv_kernels, flags, kernel_labels, fn_prefix, n_random_trials, k_max):

    last_ise = np.squeeze(ise_kernels[:, :, -1])
    last_iv  = np.squeeze(iv_kernels[ :, :, -1])

    ise_mean_log = np.squeeze(np.mean(np.log10(ise_kernels), axis=1))
    ise_std_log  = np.squeeze(np.std( np.log10(ise_kernels), axis=1))
    iv_mean_log  = np.squeeze(np.mean(np.log10(iv_kernels),  axis=1))
    iv_std_log   = np.squeeze(np.std( np.log10(iv_kernels),  axis=1))

    fig = plt.figure(figsize=(8, 6))
    plt.boxplot(last_ise.T, notch=False, labels=kernel_labels, widths=0.8)
    plt.xlabel("Gaussian Process Kernel")
    plt.yscale("log")
    plt.ylabel("Integrated Squared Error")
    plt.grid(True)
    #plt.show()

    filename = "plots/" + fn_prefix + flags[0] + "_" + flags[1] + "/kernel_comparison_ise_box_{}".format(n_random_trials)\
               + "_{}.svg".format(k_max)
    plt.savefig(filename, dpi=300)
    plt.close(fig)


    # do for iv
    fig = plt.figure(figsize=(8, 6))
    plt.boxplot(last_iv.T, notch=False, labels=kernel_labels, widths=0.8)
    plt.xlabel("Gaussian Process Kernel")
    plt.yscale("log")
    plt.ylabel("Integrated Variance")
    plt.grid(True)
    #plt.show()

    filename = "plots/" + fn_prefix + flags[0] + "_" + flags[1] + "/kernel_comparison_iv_box_{}".format(n_random_trials) \
               + "_{}.svg".format(k_max)
    plt.savefig(filename, dpi=300)
    plt.close(fig)


    # plot log trajectory over iterations with mean and sd
    fig = plt.figure(figsize=(8, 6))
    for i, label in enumerate(kernel_labels):
        plt.fill_between(range(k_max + 1),
                         np.power(10, ise_mean_log[i, :] + ise_std_log[i, :]),
                         np.power(10, ise_mean_log[i, :] - ise_std_log[i, :]),
                         alpha=0.3, lw=0)
        plt.plot(range(k_max + 1), np.power(10, ise_mean_log[i, :]), label=label)

    plt.xlim((0, k_max))
    plt.xlabel("Iteration")
    plt.yscale("log")
    plt.ylim((1e-12, 1e2))
    plt.ylabel("Integrated Squared Error")
    plt.legend()
    plt.grid(True)
    #plt.show()

    filename = "plots/" + fn_prefix + flags[0] + "_" + flags[1] + \
               "/kernel_comparison_ise_iter_{}".format(n_random_trials) + "_{}.svg".format(k_max)
    plt.savefig(filename, dpi=300)
    plt.close(fig)


    # plot log trajectory over iterations with mean and sd
    fig = plt.figure(figsize=(8, 6))
    for i, label in enumerate(kernel_labels):
        plt.fill_between(range(k_max + 1),
                         np.power(10, iv_mean_log[i, :] + iv_std_log[i, :]),
                         np.power(10, iv_mean_log[i, :] - iv_std_log[i, :]),
                         alpha=0.3, lw=0)
        plt.plot(range(k_max + 1), np.power(10, iv_mean_log[i, :]), label=label)

    plt.xlim((0, k_max))
    plt.xlabel("Iteration")
    plt.yscale("log")
    plt.ylim((1e-12, 1e2))
    plt.ylabel("Integrated Variance")
    plt.legend()
    plt.grid(True)
    #plt.show()

    filename = "plots/" + fn_prefix + flags[0] + "_" + flags[1] + \
               "/kernel_comparison_iv_iter_{}".format(n_random_trials) + "_{}.svg".format(k_max)
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def plot_activation_comparison(ise_activations, flags, activation_labels, fn_prefix, n_random_trials, k_max):

    last_ise = np.squeeze(ise_activations[:, :, -1])

    ise_mean_log = np.squeeze(np.mean(np.log10(ise_activations), axis=1))
    ise_std_log  = np.squeeze(np.std( np.log10(ise_activations), axis=1))

    fig = plt.figure(figsize=(4, 6))
    plt.boxplot(last_ise.T, notch=False, labels=activation_labels, widths=0.8)
    plt.xlabel("Activation Functions")
    plt.yscale("log")
    plt.ylabel("Integrated Squared Error")
    plt.grid(True)
    #plt.show()

    filename = "plots/" + fn_prefix + flags[0] + "_" + flags[1] + "/activation_comparison_ise_box_{}".format(n_random_trials)\
               + "_{}.svg".format(k_max)
    plt.savefig(filename, dpi=300)
    plt.close(fig)


    # plot log trajectory over iterations with mean and sd
    fig = plt.figure(figsize=(8, 6))
    for i, label in enumerate(activation_labels):
        plt.fill_between(range(k_max + 1),
                         np.power(10, ise_mean_log[i, :] + ise_std_log[i, :]),
                         np.power(10, ise_mean_log[i, :] - ise_std_log[i, :]),
                         alpha=0.3, lw=0)
        plt.plot(range(k_max + 1), np.power(10, ise_mean_log[i, :]), label=label)

    plt.xlim((0, k_max))
    plt.xlabel("Iteration")
    plt.yscale("log")
    plt.ylim((1e-12, 1e2))
    plt.ylabel("Integrated Squared Error")
    plt.legend()
    plt.grid(True)
    #plt.show()

    filename = "plots/" + fn_prefix + flags[0] + "_" + flags[1] + \
               "/activation_comparison_ise_iter_{}".format(n_random_trials) + "_{}.svg".format(k_max)
    plt.savefig(filename, dpi=300)
    plt.close(fig)