import numpy as np
import matplotlib.pyplot as plt

from .sinc_interp import sinc_interp2


def main() -> None:
    # Example script showing how to call `sinc_interp2` and visualize the result.
    
    # Coarse grid samples
    X = np.linspace(0.0, 1.0, 16, endpoint=False)
    Y = np.linspace(0.0, 1.0, 16, endpoint=False)
    XX, YY = np.meshgrid(X, Y, indexing="xy")
    f = np.cos(2 * np.pi * 3 * XX) * np.sin(2 * np.pi * 2 * YY)

    # Fine query grid
    x = np.linspace(0.0, 1.0, 64, endpoint=False)
    y = np.linspace(0.0, 1.0, 64, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    fq = sinc_interp2(f, X, Y, xx, yy)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].imshow(f, origin="lower", aspect="auto")
    ax[0].set_title("Coarse samples")
    ax[1].imshow(fq, origin="lower", aspect="auto")
    ax[1].set_title("Sinc interpolated")
    plt.show()


if __name__ == "__main__":
    main()

