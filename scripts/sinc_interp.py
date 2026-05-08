import numpy as np

def sinc_interp2(f, X, Y, x, y):
    """
    2D sinc (Nyquist) interpolation on a uniform rectangular grid.

    Parameters
    ----------
    f : array_like, shape (len(Y), len(X))
        Samples on the grid.
    X, Y : array_like, 1D
        Uniform sample coordinates along x (columns) and y (rows).
    x, y : array_like
        Query coordinates. `x` and `y` can be scalars or arrays; they will broadcast.
    """
    f, X, Y, x, y = map(np.asarray, (f, X, Y, x, y))
    dx, dy = X[1] - X[0], Y[1] - Y[0]
    Sx = np.sinc((x[..., None] - X) / dx)
    Sy = np.sinc((y[..., None] - Y) / dy)
    return np.einsum("...j,...i,ji->...", Sy, Sx, f)

