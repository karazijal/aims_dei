import numpy as np
import scipy.optimize as optimize

from tqdm.auto import tqdm

from .gp import GP, marginal_only


class optclb:
    """Helper callback for progress"""

    def __init__(self, verb=True, f=3):
        self.verb = verb
        self.f = f
        self.itr = 0

    def __call__(self, curr_x):
        x = np.exp(curr_x)
        if self.verb and self.itr % self.f == 0:
            xprint = [f"{x_:>2.3f}" for x_ in x]
            print(f"Itr {self.itr:>4d}: x=[{', '.join(xprint)}]")
        self.itr += 1


def fit_with_restarts(X, y, k, sigma, theta0, x_star, jitter=1e6, tol=1e5, n_restarts=1, theta_transform=np.exp,
                      GPimp=GP, f=3, verb=False, **kwargs):
    def obj_fun(theta):
        theta = theta_transform(theta)
        logmar = marginal_only(X, y, lambda A, B: k(A, B, *theta), sigma, jitter=jitter)
        return -logmar

    theta0_sampler = theta0
    if not callable(theta0):
        if n_restarts > 1:
            print("Fixed theta0 - will try once")
            n_restarts = 1
        theta0_sampler = lambda x: theta0

    res = []
    for r_itr in tqdm(range(n_restarts)):
        theta = theta0_sampler()
        opt_res = optimize.minimize(obj_fun, theta, method='L-BFGS-B', tol=tol, callback=optclb(verb=verb, f=f))
        res.append(opt_res)
        if verb:
            print(f"{r_itr} {opt_res.fun} {opt_res.nit} {theta_transform(opt_res.x)}")
    opt_res = min(res, key=lambda r: r.fun)
    final_theta = theta_transform(opt_res.x)
    print(f"Result: {final_theta} with {opt_res.fun} using {opt_res.nit} itrs")
    mdl = GPimp(X, y, sigma, lambda A, B: k(A, B, *final_theta), x_star, jitter=jitter, **kwargs)
    return mdl, final_theta
