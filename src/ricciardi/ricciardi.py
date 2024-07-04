from collections.abc import Callable
from typing import Union

import torch
from scipy import special
from torch import Tensor

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
        A tuple (value, None), where value is a tensor with shape
        func(broadcast(a, b)).shape. The additional None is for consistency with
        scipy.integrate.fixed_quad.

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
    def setup_context(ctx, inputs, _):
        ctx.save_for_backward(inputs[0], inputs[1])
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return (None, None, None)

        x, y = ctx.saved_tensors
        grad_x = -torch.special.erfcx(x)
        grad_y = torch.special.erfcx(y)

        # Handle the case where x or y is too large, resulting in NaN gradients
        # Note that this code is only valid in the context of the Ricciardi function
        # where we expect zero gradients when erfcx results in inf.
        grad_x = grad_x.nan_to_num(nan=torch.nan, posinf=0.0, neginf=0.0) * grad_output
        grad_y = grad_y.nan_to_num(nan=torch.nan, posinf=0.0, neginf=0.0) * grad_output

        return (grad_x, grad_y, None)


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
        n (optional): Precision level, equivalent to the order of Gauss-Legendre
          quadrature used to compute the integral of the complementary error function.
          If None, defaults to 3, 4, or 6 for input dtypes torch.bfloat16, torch.half,
          and torch.double respectively, and 5 for other (non-complex) dtypes. These
          defaults are chosen so that with the default values of sigma, tau, tau_rp,
          V_r, and theta, the output errors satisfy the default PyTorch tolerances
          for the corresponding dtype as prescribed in torch.testing.assert_close.
          Generally, n needs to be increased when V_r is smaller or theta is larger.

    Returns:
        Tensor of firing rates with shape broadcast(mu, sigma, tau, tau_rp, V_r, theta)

    Raises:
        TypeError: If mu is not a Tensor or is a complex Tensor.

    """
    if not isinstance(mu, Tensor) or torch.is_complex(mu):
        raise TypeError("mu must be a floating point or integer tensor.")

    is_floating_point = torch.is_floating_point(mu)
    dtype = mu.dtype
    if n is None:
        n = {torch.bfloat16: 3, torch.half: 4, torch.double: 6}.get(dtype, 5)

    if n > 5:
        mu = mu.double()
    elif dtype != torch.double:
        mu = mu.float()  # torch.special.erfcx does not support bfloat16 or half

    umin = (V_r - mu) / sigma
    umax = (theta - mu) / sigma

    out = (tau_rp + tau * sqrtpi * ierfcx(-umax, -umin, n=n)).reciprocal()

    if is_floating_point:
        out = out.to(dtype)

    return out
