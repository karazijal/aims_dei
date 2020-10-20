import numpy as np
from matplotlib import pyplot as plt

MAX_COND = 1e10


def _check_condition(K):
    if np.linalg.cond(K) > MAX_COND:
        print(f"k(X,X) is too close to singular {np.linalg.cond(K)}")


def _noise(X, sigmas):
    """Handle different noise for different output"""
    if isinstance(sigmas, (float, int)) or len(sigmas) == 1:
        return np.eye(len(X)) * sigmas ** 2
    else:
        return np.diag(np.array([sigmas[int(i)] ** 2 for i in X[:, 1]]))


def marginal_only(X, y, k, sigma, jitter=0):
    """
    Fit GP, returning only log marginal likelyhood
    """
    K = k(X, X) + _noise(X, sigma) + jitter * np.eye(X.shape[0])
    _check_condition(K)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(K, y)
    logmarlike = -.5 * y.T @ alpha - np.log(np.diag(L)).sum() - len(X) / 2 * np.log(2 * np.pi)
    return logmarlike


def slow(X, y, k, sigma, x_star, jitter=0):
    """
    Fit GP using matrix inverse, i.e. simpler maths.

    Ended up having to use Cholesky factor anyway as calculating determinants was returning results
    either entirely or _very_ close to zero causing `log` to blow up.
    """
    K = k(X, X) + _noise(X, sigma) + jitter * np.eye(X.shape[0])
    _check_condition(K)
    det_factor = np.log(np.diag(np.linalg.cholesky(K))).sum()
    Ki = np.linalg.pinv(K)
    Kiy = Ki @ y
    logmarlike = -.5 * (y.T @ Kiy) - det_factor - len(X) / 2 * np.log(2 * np.pi)
    fstr = k(x_star, X) @ Kiy
    covf = k(x_star, x_star) - (k(x_star, X) @ (Ki @ k(X, x_star)))
    return logmarlike, fstr, covf


def fit(X, y, k, sigmas, x_star, jitter=0):
    """Fit GP using Cholesky decom and pseudo inverse with different noise"""
    K = k(X, X) + _noise(X, sigmas) + jitter * np.eye(X.shape[0])
    _check_condition(K)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError as e:
        print(e)
        print(K)
        print(np.linalg.cond(K))
        raise
    try:
        Li = np.linalg.pinv(L)
    except np.linalg.LinAlgError as e:
        print(e)
        print(L)
        print(np.linalg.cond(L))
        raise
    alpha = Li.T @ (Li @ y)
    kstr = k(X, x_star)
    fstr = kstr.T @ alpha
    v = Li @ kstr
    covf = k(x_star, x_star) - v.T @ v
    logmarlike = -.5 * y.T @ alpha - np.log(np.diag(L)).sum() - len(X) / 2 * np.log(2 * np.pi)
    return logmarlike, fstr, covf


class GP:
    """GP implementation that expands x_star with extra points for increased resolution"""

    @staticmethod
    def _function_draw_points(X, x, npoints, expansion=.1):
        ma = X.max(axis=0)
        mi = X.min(axis=0)
        if len(x):
            ma = np.max(np.array([ma, x.max()]))
            mi = np.min(np.array([mi, x.min()]))
        amp = ma - mi
        return np.linspace(mi - amp * expansion, ma + amp * expansion, npoints)

    def __init__(self, X, y, sigma, k, x, jitter=0, npoints=1000, expansion=.1):
        self.x_extr = self._function_draw_points(X, x, npoints, expansion=expansion)
        x_star = np.concatenate((x, self.x_extr))
        self.x = x
        self.X = X
        self.y = y
        self.logmarginal, self.fstr, self.C = fit(X, y, k, sigma, x_star, jitter=jitter)

    @property
    def var(self):
        if self.C is not None:
            return np.diag(self.C)[:self.x.shape[0]]

    @property
    def f(self):
        if self.fstr is not None:
            return self.fstr[:len(self.x)]

    @property
    def f_extr(self):
        if self.fstr is not None:
            return self.fstr[len(self.x):]

    def plot_functions(self, n=1):
        if self.C is None:
            print(f"{self.method} was used")
            return
        cov = self.C[len(self.x):, len(self.x):]
        draws = np.random.multivariate_normal(self.f_extr, cov, size=n)
        std = np.sqrt(np.diag(self.C)[len(self.x):])
        plt.fill_between(self.x_extr, self.f_extr + 2 * std, self.f_extr - 2 * std, fc='lightgrey', alpha=.8,
                         label="$±2\sigma$")
        plt.fill_between(self.x_extr, self.f_extr + std, self.f_extr - std, fc='grey', alpha=.3, label="$±\sigma$")
        for i in range(n):
            plt.plot(self.x_extr, draws[i])
        plt.xlim([self.x_extr.min(), self.x_extr.max()])
        plt.title(f"{n} function draws from fitted GP posterior")

    def plot_gp(self, x_true=None, y_true=None, plot_conf=True, show_predictions=False, show_marginal=True):
        if plot_conf:
            std = np.sqrt(np.diag(self.C)[len(self.x):])
            plt.fill_between(self.x_extr, self.f_extr + 2 * std, self.f_extr - 2 * std, fc='lightgrey', alpha=.8,
                             label="$±2\sigma$")
            plt.fill_between(self.x_extr, self.f_extr + std, self.f_extr - std, fc='grey', alpha=.3, label="$±\sigma$")
        if x_true is not None and y_true is not None:
            plt.plot(x_true, y_true, 'r--', label='True $f$ (Test)')

        plt.plot(self.x_extr, self.f_extr, 'g-', label='$f_{*}$ (Predictive Post. Mean)')

        plt.plot(self.X, self.y, 'b+', label='$y(X)$ (Training)')
        if show_predictions:
            plt.plot(self.x, self.f, 'mo', label='Predictions for testing points', markerfacecolor='none')
        if len(self.x_extr):
            plt.xlim([self.x_extr.min(), self.x_extr.max()])
        plt.xlabel("Reading time (h)")
        plt.legend()
        plt.grid()
        if show_marginal:
            plt.text(0, 0.01, "$\log(p(y|X)) = {:.4f}$".format(self.logmarginal), transform=plt.gca().transAxes,
                     fontsize=16)

    def neg_log_likelyhood(self, yp):
        return (.5 * np.log(2 * np.pi * self.var) + (yp - self.f) ** 2 / 2 / self.var).sum()


class CGP(GP):
    def __init__(self, X, y, sigmas, k, x, jitter=0, npoints=1000, expansion=.1, noutputs=2):
        self.nouts = noutputs
        x_extr = self._function_draw_points(X[:, 0], x[:, 0] if len(x.shape) > 1 else x, npoints // noutputs, expansion=expansion)
        x_extr = x_extr[:, None]
        self.x_extr = [np.hstack((x_extr, np.ones_like(x_extr) * out)) for out in range(self.nouts)]
        self.x_extr = np.vstack(self.x_extr)
        x_star = np.concatenate((x, self.x_extr))
        self.x = x
        self.X = X
        self.y = y
        self.logmarginal, self.fstr, self.C = fit(X, y, k, sigmas, x_star, jitter=jitter)

    def plot_gp(self, l=0, x_true=None, y_true=None, plot_conf=True, show_predictions=False, show_marginal=False):
        mask = self.x_extr[:, 1] == l
        if plot_conf:
            std = np.sqrt(np.diag(self.C)[len(self.x):])
            plt.fill_between(self.x_extr[:, 0][mask], self.f_extr[mask] + 2 * std, self.f_extr[mask] - 2 * std,
                             fc='lightgrey', alpha=.8,
                             label="$±2\sigma$")
            plt.fill_between(self.x_extr[:, 0][mask], self.f_extr[mask] + std, self.f_extr[mask] - std, fc='grey',
                             alpha=.3, label="$±\sigma$")
        if x_true is not None and y_true is not None:
            plt.plot(x_true[:, 0][x_true[:, 1] == l], y_true[x_true[:, 1] == l], 'r--', label='True $f$ (Test)')

        plt.plot(self.x_extr[:, 0][mask], self.f_extr[mask], 'g-', label='$f_{*}$ (Predictive Post. Mean)')

        plt.plot(self.X[:, 0][self.X[:, 1] == l], self.y[self.X[:, 1] == l], 'b+', label='$y(X)$ (Training)')
        if show_predictions:
            plt.plot(self.x[:, 0][self.x[:, 1] == l], self.f[self.x[:, 1] == l], 'mo',
                     label='Predictions for testing points', markerfacecolor='none')
        if len(self.x_extr):
            plt.xlim([self.x_extr[:, 0][mask].min(), self.x_extr[:, 0][mask].max()])
        plt.xlabel("Reading time (h)")
        plt.legend()
        plt.grid()
        if show_marginal:
            plt.text(0, 0.01, "$\log(p(y|X)) = {:.4f}$".format(self.logmarginal), transform=plt.gca().transAxes,
                     fontsize=16)

    def neg_log_likelyhood(self, yp, mask=None):
        if mask is None:
            return super().neg_log_likelyhood(yp)
        return (.5 * np.log(2 * np.pi * self.var[mask]) + (yp - self.f[mask]) ** 2 / 2 / self.var[mask]).sum()
