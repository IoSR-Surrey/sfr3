import numpy as np

from scripts.sinc_interp import sinc_interp2


def test_sinc_interp2_identity_on_grid() -> None:
    # If we query exactly at the original sample points, ideal sinc interpolation must reproduce
    # the samples: sinc(0)=1 and sinc(n)=0 for any nonzero integer n, so only the (i,j) term remains.
    rng = np.random.default_rng(0)

    X = np.arange(8.0)
    Y = np.arange(6.0)
    f = rng.standard_normal((len(Y), len(X)))

    XX, YY = np.meshgrid(X, Y, indexing="xy")
    f_hat = sinc_interp2(f, X, Y, XX, YY)

    np.testing.assert_allclose(f_hat, f, rtol=0.0, atol=1e-12)


def test_sinc_interp2_delta_matches_sinc_product() -> None:
    # Put a single "1" on the grid (a 2D Kronecker delta). The interpolated field should equal the
    # kernel centered at that sample: sinc((x-x0)/dx) * sinc((y-y0)/dy).
    X = np.arange(-10.0, 11.0)
    Y = np.arange(-10.0, 11.0)

    f = np.zeros((len(Y), len(X)))
    ix, iy = 7, 12
    f[iy, ix] = 1.0

    x = np.linspace(X[0], X[-1], 401)
    y = np.linspace(Y[0], Y[-1], 401)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    fq = sinc_interp2(f, X, Y, xx, yy)
    expected = np.sinc(xx - X[ix]) * np.sinc(yy - Y[iy])

    np.testing.assert_allclose(fq, expected, rtol=0.0, atol=1e-12)


def test_sinc_interp2_agrees_with_definition() -> None:
    # Cross-check against the literal definition of 2D sinc interpolation:
    #   f̂(x,y) = Σ_j Σ_i f[j,i] * sinc((x-X[i])/dx) * sinc((y-Y[j])/dy)
    # Implemented below as explicit loops for a tiny problem (slow, but straightforward).
    rng = np.random.default_rng(1)

    X = np.arange(6.0)
    Y = np.arange(5.0)
    f = rng.standard_normal((len(Y), len(X)))

    x = np.array([0.3, 2.7, 4.1])
    y = np.array([1.2, 3.4, 0.5])

    fq = sinc_interp2(f, X, Y, x, y)

    dx, dy = X[1] - X[0], Y[1] - Y[0]
    fq_ref = np.empty_like(x, dtype=np.result_type(f, x, y))
    for k in range(len(x)):
        s = 0.0
        for j in range(len(Y)):
            for i in range(len(X)):
                s += (
                    f[j, i]
                    * np.sinc((x[k] - X[i]) / dx)
                    * np.sinc((y[k] - Y[j]) / dy)
                )
        fq_ref[k] = s

    np.testing.assert_allclose(fq, fq_ref, rtol=0.0, atol=1e-12)

