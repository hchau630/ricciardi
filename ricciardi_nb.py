import itertools

from scipy import special, integrate
from mpmath import fp
import numpy as np
import torch
import platformdirs
import joblib
import matplotlib.pyplot as plt

memory = joblib.Memory(location=platformdirs.user_cache_dir('Ricciardi', 'HYC'), verbose=0)


def factorial2(n):
    """
    See scipy.special.factorial2 for an explanation
    """
    out = 2 ** (n / 2) * (n / 2 + 1).lgamma().exp()
    return torch.where(n % 2 == 1, out * (2 / np.pi)**0.5, out)


def chebyshev(n, x):
    return torch.cos(n * torch.acos(x))


def chebyshev_np(n, x):
    return np.cos(n * np.arccos(x))


def dawson(x):
    """
    Rational approximation of Dawson's function based on
    https://www.jstor.org/stable/2004886?seq=4
    """
    absx = x.abs()
    # m0, m1, m2, m3 = absx < 2.5, (absx >= 2.5) & (absx < 3.5), (absx >= 3.5) & (absx < 5.0), absx >= 5.0
    # x0, x1, x2, x3 = x[m0], x[m1], x[m2], x[m3]
    # t0, t1, t2, t3 = x0**2, x1**2, x2**2, x3**2
    # out = torch.empty_like(x)
    
    # out[m0] = x0 * (1.08326558873e4 - 1.28405832279e3 * x0**2 + 4.19672902280e2 * x0**4 - 1.51982152422e1 * x0**6 + 1.67795116189e0 * x0**8 - 2.38594565696e-2 * x0**10) / (1.08326558772e4 + 5.93771276935e3 * x0**2 + 1.48943557242e3 * x0**4 + 2.19728331833e2 * x0**6 + 1.99422336364e1 * x0**8 + 1.0e0 * x0**10)  # rtol=1.0e-9
    # out[m1] = 1 / x1 * (5.00652754437e-1 + 2.14221965778e-1 / (-4.91605365741e0 + x1**2 + 3.73902763512e1 / (4.07068101667e0 + x1**2 + 1.38216341182e1 / (-1.26817901598e1 + x1**2 + 8.87619386764e1 / (3.47393742586e0 + x1**2)))))  # rtol=1.0e-9
    # out[m2] = 1 / x2 * (5.0000965199e-1 + 2.4924632421e-1 / (-1.5853935006e0 + x2**2 - 5.3714272981e-1 / (-1.0385024821e1 + x2**2 + 1.7761778077e1 / (-4.1083233770e0 + x2**2))))  # rtol=1.0e-8
    # out[m3] = 1 / (2 * x3) * (1 + 1 / x3**2 * (5.000381123e-1 + 7.449387745e-1 / (-2.748627766e0 + x3**2)))  # rtol=1.0e-7

    # out[m0] = x0 * (1.08326558873e4 + (-1.28405832279e3 + (4.19672902280e2 + (-1.51982152422e1 + (1.67795116189e0 - 2.38594565696e-2 * t0) * t0) * t0) * t0) * t0) / (1.08326558772e4 + (5.93771276935e3 + (1.48943557242e3 + (2.19728331833e2 + (1.99422336364e1 + t0) * t0) * t0) * t0) * t0) # rtol=1.0e-9
    # out[m1] = 1 / x1 * (5.00652754437e-1 + 2.14221965778e-1 / (-4.91605365741e0 + t1 + 3.73902763512e1 / (4.07068101667e0 + t1 + 1.38216341182e1 / (-1.26817901598e1 + t1 + 8.87619386764e1 / (3.47393742586e0 + t1)))))  # rtol=1.0e-9
    # out[m2] = 1 / x2 * (5.0000965199e-1 + 2.4924632421e-1 / (-1.5853935006e0 + t2 - 5.3714272981e-1 / (-1.0385024821e1 + t2 + 1.7761778077e1 / (-4.1083233770e0 + t2))))  # rtol=1.0e-8
    # out[m3] = 1 / (2 * x3) * (1 + 1 / t3 * (5.000381123e-1 + 7.449387745e-1 / (-2.748627766e0 + t3)))  # rtol=1.0e-7

    # return out

    t = x**2
    return torch.where(
        absx < 3.5,
        torch.where(
            absx < 2.5,
            x * (1.08326558873e4 + (-1.28405832279e3 + (4.19672902280e2 + (-1.51982152422e1 + (1.67795116189e0 - 2.38594565696e-2 * t) * t) * t) * t) * t) / (1.08326558772e4 + (5.93771276935e3 + (1.48943557242e3 + (2.19728331833e2 + (1.99422336364e1 + t) * t) * t) * t) * t),
            1 / x * (5.00652754437e-1 + 2.14221965778e-1 / (-4.91605365741e0 + t + 3.73902763512e1 / (4.07068101667e0 + t + 1.38216341182e1 / (-1.26817901598e1 + t + 8.87619386764e1 / (3.47393742586e0 + t))))),
        ),
        torch.where(
            absx < 5.0,
            1 / x * (5.0000965199e-1 + 2.4924632421e-1 / (-1.5853935006e0 + t - 5.3714272981e-1 / (-1.0385024821e1 + t + 1.7761778077e1 / (-4.1083233770e0 + t)))),
            1 / (2 * x) * (1 + 1 / t * (5.000381123e-1 + 7.449387745e-1 / (-2.748627766e0 + t))),
            # 1 / (2 * x) * (1 + 1 / t * (5.0000000167450e-1 + 7.4999919056701e-1 / (-2.5001711668562e0 + t - 2.4878765880441e0 / (-4.6731202214124e0 + t - 4.1254406560831e0 / (-1.1195216423662e1 + t))))),
        ),
    )


def dawson_approx(x):
    """
    Very low precision rational approximation
    """
    absx = x.abs()
    t = x**2
    return torch.where(
        absx < 2.5,
        x * (1.145e0 - 8.426e-2 * t) / (1.085e0 + t),
        1 / (2 * x) * (1 + 0.5 / t),
    )


def erfi(x):
    return 2 / np.pi**0.5 * torch.exp(x**2) * dawson(x)


@np.vectorize
def ierfcx_exact(x):
    return integrate.quad(special.erfcx, 0, x)[0]


def ierfcx_chebyshev(x, order=5, xt=3.25, k=3.75):
    """
    Important note: this breaks down at order 8 due to numerical imprecision
    of pytorch functions. For order 8 or more use the numpy version, which
    is much more numerically precise
    """
    if not (x >= 0.0).all() and (x < xt).all():
        raise ValueError()
    
    def F(t, min_t, max_t):
        t = 0.5 * (min_t + max_t) + 0.5 * (max_t - min_t) * t
        x = k * (1 + t) / (1 - t)
        return 1 / torch.log(1.0 + x) * torch.from_numpy(ierfcx_exact(x))
        
    n = torch.arange(order + 1, device=x.device)
    tn = torch.cos(((2 * n + 1) / (order + 1)).double() * np.pi / 2)  # Chebyshev nodes
    min_t, max_t = -1.0, (xt - k) / (xt + k)
    coef = 2 / (order + 1) * (F(tn, min_t, max_t) * chebyshev(n[:, None], tn)).sum(dim=-1)
    coef[0] = coef[0] / 2
    
    t = (x - k) / (x + k)
    t = (t - 0.5 * (min_t + max_t)) / (0.5 * (max_t - min_t))
    t = t.clamp(-1.0, 1.0)  # handle possible numerical error where t < -1.0 or t > 1.0
    return torch.log(1.0 + x) * (coef * chebyshev(n, t[:, None])).sum(dim=-1)


def ierfcx_chebyshev_np(x, order=8, xt=4.1, k=3.75):
    if not (x >= 0.0).all() and (x < xt).all():
        raise ValueError()
    
    def F(t, min_t, max_t):
        t = 0.5 * (min_t + max_t) + 0.5 * (max_t - min_t) * t
        x = k * (1 + t) / (1 - t)
        return 1 / np.log(1.0 + x) * ierfcx_exact(x)
        
    n = np.arange(order + 1)
    tn = np.cos((2 * n + 1) / (order + 1) * np.pi / 2)  # Chebyshev nodes
    min_t, max_t = -1.0, (xt - k) / (xt + k)
    coef = 2 / (order + 1) * (F(tn, min_t, max_t) * chebyshev_np(n[:, None], tn)).sum(axis=-1)
    coef[0] = coef[0] / 2
    chebyshev = np.polynomial.Chebyshev(coef, domain=(min_t, max_t))
    
    # with np.printoptions(formatter={'float': '{:.12e}'.format}):
    #     print(chebyshev.convert(kind=np.polynomial.Polynomial).coef)

    t = (x - k) / (x + k)
    return np.log(1.0 + x) * chebyshev(t)


def ierfcx_asymptotic(x, order=8, xt=4.1):
    if not (x >= xt).all():
        raise ValueError()
    
    n = torch.arange(1, order, device=x.device)
    
    def f(x):
        return 1 / np.pi**0.5 * (torch.log(x) - ((-1)**n * factorial2(2 * n - 1) / (2 * n * (2 * x[..., None]**2)**n)).sum(dim=-1))

    # with np.printoptions(formatter={'float': '{:.12e}'.format}):
    #     print(ierfcx_exact(xt)[None] - f(torch.tensor(xt).double()).numpy()[None])
    
    return f(x) - f(torch.tensor(xt).double()) + ierfcx_exact(xt)

    
def ierfcx(x, c=8, a=8, xt=4.1, k=3.75):
    """
    Approximation of the integral of erfcx(t) = exp(t^2) * erfc(t)
    from 0 to x. For x >= xt, the integral is approximated by
    integrating the asymptotic expansion of erfcx(t). For 0 <= x < xt,
    first define f(x) = 1 / log(1 + x) * \int_0^x erfcx(t) dt,
    then define F(t) = f(k * (1 + t) / (1 - t)). F(t) is approximated by
    chebyshev interpolation for -1 <= t < (xt - k) / (xt + k). The
    integral is then approximated as log(1 + x) * F((x - k) / (x + k)).
    A similar approximation procedure used for computing erfcx(x)
    is detailed in https://www.jstor.org/stable/2007742?seq=1.
    For x < 0, we use the fact that erfc(-t) = 2 - erfc(t) to compute the
    integral by writing
    
    \int_0^{-x} erfcx(t) dt
    = -\int_0^x erfcx(-t) dt
    = \int_0^x erfcx(t) dt - 2\int_0^x exp(t^2) dt
    = \int_0^x erfcx(t) dt - \sqrt{\pi} erfi(x)

    The default parameters are chosen so that the relative error is less than 1.0e-7.

    Args:
        c: order of the interpolating Chebyshev polynoomial
        a: order (number of terms) in the asymptotic expansion
        xt: threshold of x for transition from Chebyshev to asymptotic approx
        k: parameter of the mapping of x \in [0, \infty) to t \in [-1, 1)
    Returns:
        Approximation of \int_0^x erfcx(t) dt
    """
    absx = x.abs()
    mask_n, mask_c, mask_a = x < 0, absx < xt, absx >= xt
    out = torch.empty_like(x)
    # out[mask_c] = ierfcx_chebyshev(absx[mask_c], order=c, xt=xt, k=k).to(x.dtype)  # not enough precision
    out[mask_c] = torch.from_numpy(ierfcx_chebyshev_np(absx[mask_c].numpy(), order=c, xt=xt, k=k)).to(x.dtype)
    out[mask_a] = ierfcx_asymptotic(absx[mask_a], order=a, xt=xt)
    out[mask_n] = out[mask_n] - np.pi ** 0.5 * erfi(absx[mask_n])
    return out


def ierfcx_chebyshev_fast(x):
    t = (x - 3.75) / (x + 3.75)
    return torch.log(1.0 + x) * (8.402081835053e-01 + (-1.506407525002e-01 + (1.611762752505e-02 + (-1.336764713619e-02 + (-1.433259084023e-02 + (-1.050071231432e-02 + (-2.547932936248e-02 + (-1.672446952458e-02 - 7.747389892958e-03 * t) * t) * t) * t) * t) * t) * t) * t)


def ierfcx_asymptotic_fast(x):
    t = x**-2
    return 1 / np.pi**0.5 * (torch.log(x) + (0.25 + (-0.1875 + (0.3125 + (-0.8203125 + (2.953125 + (-13.53515625 + 75.41015625 * t) * t) * t) * t) * t) * t) * t) + 5.538959341195e-01


def ierfcx_fast(x):
    absx = x.abs()
    out = torch.where(absx >= 4.1, ierfcx_asymptotic_fast(absx), ierfcx_chebyshev_fast(absx))
    neg = x < 0
    out[neg] -= np.pi**0.5 * erfi(absx[neg])
    # out = torch.where(x >= 0, out, out - np.pi**0.5 * erfi(absx))
    return out


def ierfcx_approx(x):
    ax = x.abs()
    out = torch.log(1.0 + ax) * (8.378565612390e-01 - 1.657794076194e-01 * (ax - 3.75) / (ax + 3.75))
    out = torch.where(x >= 0, out, out - 2 * torch.exp(ax**2) * dawson_approx(ax))
    return out


def ricciardi(mu, tau=0.02, tau_rp=0.002, sigma_t=0.01, V_r=0.01, theta=0.02, **kwargs):
    mu = mu.double()
    min_u = (V_r - mu) / sigma_t
    max_u = (theta - mu) / sigma_t
    # return 1.0 / (tau_rp + tau * np.sqrt(np.pi) * (ierfcx(-min_u, **kwargs) - ierfcx(-max_u, **kwargs)))
    mask = -min_u > -10
    out = torch.empty_like(mu)
    out[mask] = 1.0 / (tau_rp + tau * np.sqrt(np.pi) * (ierfcx(-min_u[mask]) - ierfcx(-max_u[mask])))
    u = max_u[~mask]
    out[~mask] = u * torch.exp(-u**2) / (tau * np.sqrt(np.pi))  # avoid NaNs, can't use torch.where due to NaN gradient
    return out


def ricciardi_fast(mu, tau=0.02, tau_rp=0.002, sigma_t=0.01, V_r=0.01, theta=0.02):
    mu = mu.double()
    min_u = (V_r - mu) / sigma_t
    max_u = (theta - mu) / sigma_t
    if (-min_u).min() > -10:
        return 1.0 / (tau_rp + tau * np.sqrt(np.pi) * (ierfcx_fast(-min_u) - ierfcx_fast(-max_u)))
    
    mask = -min_u > -10
    out = torch.empty_like(mu)
    out[mask] = 1.0 / (tau_rp + tau * np.sqrt(np.pi) * (ierfcx_fast(-min_u[mask]) - ierfcx_fast(-max_u[mask])))
    u = max_u[~mask]
    out[~mask] = u * torch.exp(-u**2) / (tau * np.sqrt(np.pi))  # avoid NaNs, can't use torch.where due to NaN gradient
    return out


def ricciardi_approx(mu, tau=0.02, tau_rp=0.002, sigma_t=0.01, V_r=0.01, theta=0.02):
    min_u = (V_r - mu) / sigma_t
    max_u = (theta - mu) / sigma_t
    if (-min_u).min() > -10:
        return 1.0 / (tau_rp + tau * np.sqrt(np.pi) * (ierfcx_approx(-min_u) - ierfcx_approx(-max_u)))
    
    mask = -min_u > -10
    out = torch.empty_like(mu)
    out[mask] = 1.0 / (tau_rp + tau * np.sqrt(np.pi) * (ierfcx_approx(-min_u[mask]) - ierfcx_approx(-max_u[mask])))
    u = max_u[~mask]
    out[~mask] = u * torch.exp(-u**2) / (tau * np.sqrt(np.pi))  # avoid NaNs, can't use torch.where due to NaN gradient
    return out


@np.vectorize
def ricciardi_np(mu, tau, tau_rp, sigma_t, V_r, theta, approx=True):
    """
    Parameters have the same meaning as defined in this paper by Alessandro
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008165
    """
    min_u = (V_r - mu) / sigma_t
    max_u = (theta - mu) / sigma_t

    if not approx:
        # Note: scipy.special.erfcx(-x) = e^{x^2}erfc(-x) = e^{x^2}(1 + erf(x))
        r = 1.0 / (tau_rp + tau * np.sqrt(np.pi) * integrate.quad(special.erfcx, -max_u, -min_u)[0])

    elif min_u > 10:
        r = max_u / tau / np.sqrt(np.pi) * np.exp(-max_u**2)
        
    elif max_u > -3.7:
        r = 1.0 / (tau_rp + tau * (0.5 * np.pi * (special.erfi(max_u) - special.erfi(min_u)) +
                   max_u**2 * fp.hyp2f2(1.0, 1.0, 1.5, 2.0, max_u**2) -
                   min_u**2 * fp.hyp2f2(1.0, 1.0, 1.5, 2.0, min_u**2)))
        
    else:
        r = 1.0 / (tau_rp + tau * (np.log(abs(min_u)) - np.log(abs(max_u)) +
                   (0.25 * min_u**-2 - 0.1875 * min_u**-4 + 0.3125 * min_u**-6 -
                    0.8203125 * min_u**-8 + 2.953125 * min_u**-10 - 13.53515625 * min_u**-12 + 75.41015625 * min_u**-14) -
                   (0.25 * max_u**-2 - 0.1875 * max_u**-4 + 0.3125 * max_u**-6 -
                    0.8203125 * max_u**-8 + 2.953125 * max_u**-10 - 13.53515625 * max_u**-12 + 75.41015625 * max_u**-14)))

    return r


@memory.cache
def ricciardi_table(tau, tau_rp, sigma_t, V_r, theta, approx=True):
    mu = np.r_[-1.0e4, np.linspace(-2.0, 10.0, 240001), 1.0e4]  # step-size of 5.0e-5 between -2 and 10
    return mu, ricciardi_np(mu, tau, tau_rp, sigma_t, V_r, theta, approx=approx)


class Ricciardi(torch.nn.Module):
    def __init__(self, tau=0.02, tau_rp=0.002, sigma_t=0.01, V_r=0.01, theta=0.02, approx=True):
        super().__init__()
        
        points, values = ricciardi_table(tau, tau_rp, sigma_t, V_r, theta, approx=approx)
        points, values = torch.from_numpy(points).float(), torch.from_numpy(values).float()
        points, values = torch.nn.Parameter(points, requires_grad=False), torch.nn.Parameter(values, requires_grad=False)
        self._interpolator = RegularGridInterpolator((points,), values)

    def forward(self, x):
        return self._interpolator(x[None, ...])


class RegularGridInterpolator(torch.nn.Module):
    def __init__(self, points, values):
        if not isinstance(points, (tuple, list)) or not all(isinstance(p, torch.Tensor) and p.ndim == 1 for p in points):
            raise TypeError("points must be a tuple or list of 1-dimensional torch.Tensor.")

        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor.")

        if tuple(len(p) for p in points) != values.shape:
            raise ValueError("shape of values is incompatible with points.")
        
        super().__init__()
        
        self.points = torch.nn.ParameterList(points)
        self.values = values
        self.ndim = len(points)

    def forward(self, x):
        if len(x) != self.ndim:
            raise ValueError(f"length of x, {len(x)}, must be equal to the number of grid dimensions, {self.ndim}.")
            
        idxs = []
        dists = []
        overalls = []
        for p, xi in zip(self.points, x):
            idx_right = torch.bucketize(xi, p)
            idx_right[idx_right >= p.shape[0]] = p.shape[0] - 1
            idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
            dist_left = xi - p[idx_left]
            dist_right = p[idx_right] - xi
            dist_left[dist_left < 0] = 0.
            dist_right[dist_right < 0] = 0.
            both_zero = (dist_left == 0) & (dist_right == 0)
            dist_left[both_zero] = dist_right[both_zero] = 1.

            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        numerator = 0.
        for indexer in itertools.product([0, 1], repeat=self.ndim):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
            numerator += self.values[as_s] * torch.stack(bs_s).prod(dim=0)
        denominator = torch.stack(overalls).prod(dim=0)
        return numerator / denominator
        