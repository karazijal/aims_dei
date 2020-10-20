import numpy as np


def rbf(A, B, alpha, l):
    A = np.atleast_1d(A)
    B = np.atleast_1d(B)
    D = (A[:, None] - B[None, :]) ** 2
    return alpha ** 2 * np.exp(-D / (2 * l ** 2))


def periodic(A, B, alpha, l, p):
    A = np.atleast_1d(A)
    B = np.atleast_1d(B)
    D = np.abs(A[:, None] - B[None, :])
    P = np.sin(np.pi * D / p) ** 2
    return alpha ** 2 * np.exp(-2 * P / (l ** 2))


def linear(A, B, alpha, c):
    A = np.atleast_1d(A)
    B = np.atleast_1d(B)
    return alpha ** 2 * (A - c)[:, None] * (B - c)[None, :]


def _corcov(A, B, W, kappa):
    "A, B are include index l in position 1. Returns Covariance matrix"
    L = W[:, None] @ W[:, None].T + np.diag(kappa)
    # inds = np.array(np.meshgrid(A[:, 1].astype(int), B[:, 1].astype(int)))
    # K_l = np.zeros(inds.shape[1:])
    # for i, j in np.ndindex(6,6):
    #   r,c = inds[:, i,j]
    #   K_l[i, j] = L[r,c]
    # return K_l.T
    # Equivalent to the above but faster
    return np.take(np.take(L, A[:, 1].astype(int), axis=0).T, B[:, 1].astype(int), axis=0).T


def _dimpicker(k):
    def cov(A, B, *theta):
        return k(A[:, 0], B[:, 0], *theta)

    return cov


def coregionalise(k, noutputs=2):
    """Co-regionalises cov. function k"""
    K = _dimpicker(k)

    def _wrapper(A, B, *params):
        W = np.array(params[:noutputs])
        kappa = np.array(params[noutputs:2 * noutputs])
        theta = params[2 * noutputs:]
        K_l = _corcov(A, B, W, kappa)
        return K_l * K(A, B, *theta)

    return _wrapper


if __name__ == '__main__':
    A = np.random.rand(10)
    B = np.random.rand(2)
    assert np.all(rbf(A, B, 1, 1) == np.atleast_2d(np.array([[rbf(a, b, 1, 1) for b in B] for a in A]).squeeze()))

    A = B = np.ones(10)
    assert np.all(np.ones((10, 10)) == rbf(A, B, 1, 1000))

    A = np.zeros(10)
    B = np.ones(10)
    assert np.all(np.ones((10, 10)) == periodic(A, B, 1, 1, 1))
