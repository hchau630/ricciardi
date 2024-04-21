import torch
from scipy import special

sqrtpi = torch.pi**0.5


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
    """Like scipy.integrate.fixed_quad, but for tensor inputs a, b

    Args:
        func (Callable[[torch.Tensor, ...], torch.Tensor]): Integrand.
        a: (torch.Tensor): Lower limits of integral.
        b: (float | torch.Tensor): Upper limit(s) of integral.
        args (tuple, optional): Additional arguments passed to func.
        n (int, optional): Order of Gauss-Legendre quadrature.

    Returns:
        torch.Tensor: Tensor with shape func(broadcast(a, b)).shape

    """
    x, w = _cached_roots_legendre(n)
    x = x.real

    d = (b - a) / 2.0
    return d * sum(wi * func(d * (xi + 1) + a, *args) for xi, wi in zip(x, w)), None


class Ierfcx(torch.autograd.Function):
    @staticmethod
    def forward(x, y, n):
        return fixed_quad(torch.special.erfcx, x, y, n=n)[0]

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], inputs[1])

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return (
            -torch.special.erfcx(x) * grad_output,
            torch.special.erfcx(y) * grad_output,
            None,
        )


def ierfcx(x, y, n=5):
    """Integral of erfcx(t) from x to y using Gauss-Legendre quadrature of order n.

    Args:
        x (torch.Tensor): Lower limits of integral.
        y (float | torch.Tensor): Upper limit(s) of integral.
        n (int, optional): Order of Gauss-Legendre quadrature.

    Returns:
        torch.Tensor: Tensor with shape broadcast(x, y).shape

    """
    return Ierfcx.apply(x, y, n)


def ricciardi(mu, sigma=0.01, tau=0.02, tau_rp=0.002, V_r=0.01, theta=0.02):
    """Ricciardi transfer function.

    Computes the firing rate of an LIF neuron as a function of the mean and variance
    of the presynaptic input current, with default values and notation following
    Sanzeni et al. (2020). Default values assume SI units (i.e. seconds and volts).

    Args:
        mu (torch.Tensor): Mean of presynaptic input current.
        sigma (float | torch.Tensor, optional): Standard deviation of presynaptic input current.
        tau (float | torch.Tensor, optional): Time constant of the membrane potential.
        tau_rp (float | torch.Tensor, optional): Refractory period.
        V_r (float | torch.Tensor, optional): Reset membrane potential.
        theta (float | torch.Tensor, optional): Firing threshold membrane potential.

    Returns:
        torch.Tensor: Tensor of firing rates with shape broadcast(mu, sigma, tau, tau_rp, V_r, theta).shape

    """
    u_min = (V_r - mu) / sigma
    u_max = (theta - mu) / sigma

    if (-u_min).min() > -10:
        # slightly faster path when there is no extreme value
        out = 1.0 / (tau_rp + tau * sqrtpi * ierfcx(-u_max, -u_min))
    else:
        # Handle extreme values separately to avoid numerical issues
        mask = -u_min > -10
        out = torch.empty_like(mu)
        out[mask] = 1.0 / (tau_rp + tau * sqrtpi * ierfcx(-u_max[mask], -u_min[mask]))
        u = u_max[~mask]
        out[~mask] = u * torch.exp(-(u**2)) / (tau * sqrtpi)

    return out
