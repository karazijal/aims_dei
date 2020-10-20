import jax.numpy as npj

MAX_COND = 1e10


def _check_condition(K):
    if npj.linalg.cond(K) > MAX_COND:
        print(f"k(X,X) is too close to singular {npj.linalg.cond(K)}")


def _noise(X, sigmas):
    """Handle different noise for different output"""
    if isinstance(sigmas, (float, int)) or len(sigmas) == 1:
        return npj.eye(len(X)) * sigmas ** 2
    else:
        return npj.diag(npj.array([sigmas[int(i)] ** 2 for i in X[:, 1]]))


def jax_marginal_only(X, y, k, sigma, jitter=0):
    """
    Fit GP, returning only log marginal likelyhood
    """
    K = k(X, X) + _noise(X, sigma) + jitter * npj.eye(X.shape[0])
    _check_condition(K)
    L = npj.linalg.cholesky(K)
    alpha = npj.linalg.solve(K, y)
    logmarlike = -.5 * y.T @ alpha - npj.log(npj.diag(L)).sum() - len(X) / 2 * npj.log(2 * npj.pi)
    return logmarlike