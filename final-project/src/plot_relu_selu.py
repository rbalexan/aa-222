import numpy as np
import matplotlib.pyplot as plt
from sampling import ndgrid


fig, axs = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
padding = 0.1

a = [-2]
b = [2]

X = ndgrid(a, b, m=501)

y_relu = np.maximum(0, X)
y_selu = (np.exp(X) - 1)*(X < 0) + X*(X >= 0)

x_lims = (a[0], b[0])
y_lims = (-1, 2)
x_w = x_lims[1] - x_lims[0]
y_w = y_lims[1] - y_lims[0]

i = 0

for j in range(2):

    # plot function
    if j == 0:
        axs[i, j].plot(X, y_relu, c=(255 / 255, 77 / 255, 113 / 255), lw=2, label="ReLU")
    elif j == 1:
        axs[i, j].plot(X, y_selu, c=(237 / 255, 231 / 255, 45 / 255), lw=2, label="SELU")

    axs[i, j].set_xlim((x_lims[0] - padding * x_w, x_lims[1] + padding * x_w))
    axs[i, j].set_ylim((y_lims[0] - padding * y_w, y_lims[1] + padding * y_w))
    axs[i, j].set_xlabel("z")
    axs[i, j].set_ylabel("σ(z)")
    axs[i, j].set_aspect("equal")
    axs[i, j].grid(True)
    axs[i, j].legend()

    j += 1

plt.tight_layout()
plt.savefig("plots/activation_functions.svg", dpi=300)
plt.show()
plt.close(fig)
