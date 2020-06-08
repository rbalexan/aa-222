import numpy as np
import matplotlib.pyplot as plt
from sampling import ndgrid
from benchmark_functions import sinc, hebbal, step, problem15, problem20

functions = [hebbal, problem15, problem20, sinc, sinc, step]
lims = [(0, 1), (-5, 5), (-5, 5), (-5, 5), (-5, 15), (-2, 2)]
titles = ["Hebbal", "Problem15", "Problem20", "Sinc", "SincShifted", "ZeroMeanStep"]

fig, axs = plt.subplots(2, 3, figsize=(12, 6), squeeze=False)
padding = 0.1

i = 0
j = 0

for fi, f in enumerate(functions):

    axs[i, j].grid(True)
    a, b = lims[fi]
    a = [a]
    b = [b]
    X = ndgrid(a, b, m=501)
    y = f(X)

    # plot function
    axs[i, j].plot(X, y, c=(52/255, 122/255, 235/255))
    axs[i, j].scatter(X[0], y[0], c="black")
    axs[i, j].scatter(X[-1], y[-1], c="black")

    # format plot
    x_lims = (a[0], b[0])
    y_lims = (min(y), max(y))
    x_w = x_lims[1] - x_lims[0]
    y_w = y_lims[1] - y_lims[0]

    axs[i, j].set_xlim((x_lims[0] - padding * x_w, x_lims[1] + padding * x_w))
    axs[i, j].set_ylim((y_lims[0] - padding * y_w, y_lims[1] + padding * y_w))
    axs[i, j].set_xlabel("x")
    axs[i, j].set_ylabel("y")

    axs[i, j].set_title(titles[fi], fontsize=11)

    if j != 2:
        j += 1
    elif j == 2:
        j = 0
        i += 1


plt.tight_layout()
plt.savefig("plots/test_functions.svg", dpi=300)
plt.show()
plt.close(fig)