import matplotlib.pyplot as plt
import numpy as np


def haar(ts):
    ys = np.ones_like(ts)
    ys[ts < 0] = 0
    ys[ts > 1] = 0
    ys[(ts >= 0) & (ts < 0.5)] = 1
    ys[(ts >= 0.5) & (ts < 1)] = -1

    return ys


def main():
    xs = np.linspace(0, 1, 1024)
    ys = np.cos(xs**2)
    # use the Haar wavelet to decompose the signal

    plt.plot(xs, ys)


if __name__ == "__main__":
    main()
    main()
