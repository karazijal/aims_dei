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
