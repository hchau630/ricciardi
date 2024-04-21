from scipy import special


def _cached_roots_legendre(n):
    """
    Cache roots_legendre results to speed up calls of the fixed_quad
    function.
    """
    if n in _cached_roots_legendre.cache:
        return _cached_roots_legendre.cache[n]

    _cached_roots_legendre.cache[n] = special.roots_legendre(n)
    return _cached_roots_legendre.cache[n]


_cached_roots_legendre.cache = dict()


def fixed_quad(func, a, b, args=(), n=5):
    """
    Like scipy.integrate.fixed_quad, but for tensor inputs a, b

    Args:
        func: Callable[[torch.Tensor, ...], torch.Tensor]. Integrand.
        a, b: torch.Tensor. Lower and upper limits of integral.
        args (optional): tuple. Additional arguments passed to func.
        n (optional): int. Order of Gauss-Legendre quadrature.
    """
    x, w = _cached_roots_legendre(n)
    x = x.real

    d = (b - a) / 2.0
    return d * sum(wi * func(d * (xi + 1) + a, *args) for xi, wi in zip(x, w)), None


def newton_cotes(f, a, b, n=4, kind="closed"):
    if kind == "closed":
        h = (b - a) / n
        if n == 1:
            out = 1 / 2 * h * (f(a) + f(b))
        elif n == 2:
            out = 1 / 3 * h * (f(a) + 4 * f(a + h) + f(b))
        elif n == 3:
            out = 3 / 8 * h * (f(a) + 3 * f(a + h) + 3 * f(b - h) + f(b))
        elif n == 4:
            out = (
                2
                / 45
                * h
                * (
                    7 * f(a)
                    + 32 * f(a + h)
                    + 12 * f(a + 2 * h)
                    + 32 * f(b - h)
                    + 7 * f(b)
                )
            )
        else:
            raise NotImplementedError()
    elif kind == "open":
        if n == 0:
            out = 2 * h * f(a + b / 2)
        else:
            raise NotImplementedError()

    return out
