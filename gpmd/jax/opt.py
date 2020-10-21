import numpy as np
import jax.numpy as npj
from jax import value_and_grad
import scipy.optimize as optimize
from scipy.optimize.lbfgsb import fmin_l_bfgs_b

from tqdm.auto import tqdm

from ..opt import optclb
from ..gp import GP
from .gp import jax_marginal_only


class gradstorehelper:
    def __init__(self, fun):
        self.f = fun

    def __call__(self, *args, **kwargs):
        r, self.g = self.f(*args, **kwargs)
        return r

    def grad(self, *args, **kwargs):
        return self.g


def fit_with_restarts(X, y, k, sigma, theta0, x_star, jitter=1e6, tol=1e5, n_restarts=1, theta_transform=npj.exp,
                      GPimp=GP, f=3, verb=False, **kwargs):
    @gradstorehelper
    @value_and_grad
    def obj_fun(theta):
        theta = theta_transform(theta)
        logmar = jax_marginal_only(X, y, lambda A, B: k(A, B, *theta), sigma, jitter=jitter)
        return -logmar

    theta0_sampler = theta0
    if not callable(theta0):
        if n_restarts > 1:
            print("Fixed theta0 - will try once")
            n_restarts = 1
        theta0_sampler = lambda: theta0

    res = []
    for r_itr in tqdm(range(n_restarts)):
        theta = theta0_sampler()
        opt_res = optimize.minimize(obj_fun, theta, method='L-BFGS-B', tol=tol, callback=optclb(verb=verb, f=f),
                                    jac=obj_fun.grad)
        res.append(opt_res)
        if verb:
            print(f"{r_itr} {opt_res.fun} {opt_res.nit} {theta_transform(opt_res.x)}")
    opt_res = min(res, key=lambda r: r.fun)
    final_theta = theta_transform(opt_res.x)
    print(f"Result: {final_theta} with {opt_res.fun} using {opt_res.nit} itrs")
    mdl = GPimp(X, y, sigma, lambda A, B: k(A, B, *final_theta), x_star, jitter=jitter, **kwargs)
    return mdl, final_theta


def fit_with_restarts2(X, y, k, sigma, theta0, x_star, jitter=1e6, tol=1e5, n_restarts=1, theta_transform=npj.exp,
                      GPimp=GP, f=3, verb=False, **kwargs):
    @value_and_grad
    def obj_fun(theta):
        theta = theta_transform(theta)
        logmar = jax_marginal_only(X, y, lambda A, B: k(A, B, *theta), sigma, jitter=jitter)
        return -logmar

    def fun(theta):
        f, df =obj_fun(theta)
        return np.array(f), np.array(df)

    theta0_sampler = theta0
    if not callable(theta0):
        if n_restarts > 1:
            print("Fixed theta0 - will try once")
            n_restarts = 1
        theta0_sampler = lambda: theta0

    res = []
    for r_itr in tqdm(range(n_restarts)):
        theta = theta0_sampler()
        theta, minnegmarg, info = fmin_l_bfgs_b(fun, theta, disp=True, callback=optclb(verb=verb, f=f), approx_grad=False, fprime=None)
        res.append((theta, minnegmarg, info))
        if verb:
            print(f"{r_itr} {minnegmarg} {info['nit']} {theta_transform(theta)}")
    opt_res = min(res, key=lambda r: r[1])
    final_theta = theta_transform(opt_res[0])
    print(f"Result: {final_theta} with {opt_res[1]} using {opt_res[2]['nit']} itrs")
    mdl = GPimp(X, y, sigma, lambda A, B: k(A, B, *final_theta), x_star, jitter=jitter, **kwargs)
    return mdl, final_theta


def fit_gd(X, y, k, sigma, theta0, x_star, jitter=1e6, tol=1e5, n_restarts=1, nitr=1000, lr=1e-3,
                                theta_transform=npj.exp,
                                GPimp=GP, f=3, verb=False, **kwargs):
    @value_and_grad
    def obj_fun(theta):
        theta = theta_transform(theta)
        logmar = jax_marginal_only(X, y, lambda A, B: k(A, B, *theta), sigma, jitter=jitter)
        return -logmar

    res = []
    theta0_sampler = theta0
    if not callable(theta0):
        if n_restarts > 1:
            print("Fixed theta0 - will try once")
            n_restarts = 1
        theta0_sampler = lambda: theta0

    res = []
    for r_itr in tqdm(range(n_restarts)):
        theta = theta0_sampler()
        l = 0
        for gdi in tqdm(range(nitr)):
            nl, dx = obj_fun(theta)
            if npj.any(npj.isnan(dx)):
                print(nl, theta_transform(theta), dx)
            theta -= lr * dx
            deltal = l - nl
            l = nl
            if npj.abs(deltal) < tol:
                break
            elif r_itr % 10 == 0 and verb:
                print(r_itr, nl, theta_transform(theta))
        else:
            print(f"Terminating with {deltal}")
        res.append((theta, nl, r_itr))
        if verb:
            print(f"{r_itr} {nl} {r_itr} {theta_transform(theta)}")
    opt_res = min(res, key=lambda r: r[1])
    final_theta = theta_transform(opt_res[0])
    print(f"Result: {final_theta} with {opt_res[1]} using {opt_res[2]} itrs")
    mdl = GPimp(X, y, sigma, lambda A, B: k(A, B, *final_theta), x_star, jitter=jitter, **kwargs)
    return mdl, final_theta
