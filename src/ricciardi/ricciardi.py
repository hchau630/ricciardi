from collections.abc import Callable
from typing import Union

import torch
from torch import Tensor
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


def fixed_quad(
    func: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Union[float, Tensor],
    args: tuple = (),
    n: int = 5,
) -> tuple[Tensor, None]:
    """Like scipy.integrate.fixed_quad, but for tensor inputs a, b

    Args:
        func: Integrand.
        a: Lower limits of integral.
        b: Upper limit(s) of integral.
        args (optional): Additional arguments passed to func.
        n (optional): Order of Gauss-Legendre quadrature.

    Returns:
        A tuple (value, None), where value is a tensor with shape func(broadcast(a, b)).shape.
        The additional None is for consistency with scipy.integrate.fixed_quad.

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


def ierfcx(x: Tensor, y: Union[float, Tensor], n: int = 5) -> Tensor:
    """Integral of erfcx(t) from x to y using Gauss-Legendre quadrature of order n.

    Args:
        x: Lower limits of integral.
        y: Upper limit(s) of integral.
        n (optional): Order of Gauss-Legendre quadrature.

    Returns:
        Tensor with shape broadcast(x, y).shape

    """
    return Ierfcx.apply(x, y, n)


def ricciardi(
    mu: Tensor,
    sigma: Union[float, Tensor] = 0.01,
    tau: Union[float, Tensor] = 0.02,
    tau_rp: Union[float, Tensor] = 0.002,
    V_r: Union[float, Tensor] = 0.01,
    theta: Union[float, Tensor] = 0.02,
    n: Union[int, None] = None,
) -> Tensor:
    """Ricciardi transfer function.

    Computes the firing rate of an LIF neuron as a function of the mean and variance
    of the presynaptic input current, with default values and notation following
    Sanzeni et al. (2020). Default values assume SI units (i.e. seconds and volts).

    Args:
        mu: Mean of presynaptic input current.
        sigma (optional): Standard deviation of presynaptic input current.
        tau (optional): Time constant of the membrane potential.
        tau_rp (optional): Refractory period.
        V_r (optional): Reset membrane potential.
        theta (optional): Firing threshold membrane potential.
        n (optional): Precision level, roughly equivalent to the order of Gauss-Legendre
          quadrature used to compute the integral of the complementary error function.
          If None, defaults to 3, 4, 5, or 6 for input dtypes torch.bfloat16, torch.half,
          torch.float, and torch.double, respectively.

    Returns:
        Tensor of firing rates with shape broadcast(mu, sigma, tau, tau_rp, V_r, theta).shape

    """
    if not isinstance(mu, Tensor) or not torch.is_floating_point(mu):
        raise TypeError("mu must be a floating point tensor.")

    dtype = mu.dtype
    if n is None:
        n = {torch.bfloat16: 3, torch.half: 4, torch.float: 5, torch.double: 6}[dtype]

    if n > 5:
        mu = mu.double()
    elif dtype in {torch.bfloat16, torch.half}:
        mu = mu.float()  # torch.special.erfcx does not support bfloat16 or half

    umin = (V_r - mu) / sigma
    umax = (theta - mu) / sigma

    if (-umin).min() > -10:
        # slightly faster path when there is no extreme value
        out = 1 / (tau_rp + tau * sqrtpi * ierfcx(-umax, -umin, n=n))
    else:
        # Handle extreme values separately to avoid numerical issues
        mask = -umin > -10
        out = torch.empty_like(mu)
        out[mask] = 1 / (tau_rp + tau * sqrtpi * ierfcx(-umax[mask], -umin[mask], n=n))
        u = umax[~mask]
        out[~mask] = u * torch.exp(-(u**2)) / (tau * sqrtpi)

    return out.to(dtype)
