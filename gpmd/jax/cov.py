import jax.numpy as npj


def jax_rbf(A, B, sigma, l):
    A = npj.atleast_1d(A)
    B = npj.atleast_1d(B)
    D = (A[:, None] - B[None, :]) ** 2
    return sigma ** 2 * npj.exp(-D / (2 * l ** 2))


def jax_periodic(A, B, sigma, l, p):
    A = npj.atleast_1d(A)
    B = npj.atleast_1d(B)
    D = npj.abs(A[:, None] - B[None, :])
    P = npj.sin(npj.pi * D / p) ** 2
    return sigma ** 2 * npj.exp(-2 * P / (l ** 2))

def _corcov(A, B, W, kappa):
    "A, B are include index l in position 1. Returns Covariance matrix"
    L = W[:, None] @ W[:, None].T + npj.diag(kappa)
    # inds = np.array(np.meshgrid(A[:, 1].astype(int), B[:, 1].astype(int)))
    # K_l = np.zeros(inds.shape[1:])
    # for i, j in np.ndindex(6,6):
    #   r,c = inds[:, i,j]
    #   K_l[i, j] = L[r,c]
    # return K_l.T
    # Equivalent to the above but faster
    return npj.take(npj.take(L, A[:, 1].astype(int), axis=0).T, B[:, 1].astype(int), axis=0).T


def _dimpicker(k):
    def cov(A, B, *theta):
        return k(A[:, 0], B[:, 0], *theta)

    return cov


def jax_coregionalise(k, noutputs=2):
    """Co-regionalises cov. function k"""
    K = _dimpicker(k)

    def _wrapper(A, B, *params):
        W = npj.array(params[:noutputs])
        kappa = npj.array(params[noutputs:2 * noutputs])
        theta = params[2 * noutputs:]
        K_l = _corcov(A, B, W, kappa)
        return K_l * K(A, B, *theta)

    return _wrapper