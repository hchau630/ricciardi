import functools

import numpy as np
from scipy import integrate, special
import torch
import pytest

from ricciardi import ricciardi


@pytest.fixture
def params():
    return dict(tau=0.02, tau_rp=0.002, sigma=0.01, V_r=0.01, theta=0.02)


@np.vectorize
def ierfcx_exact(x, y):
    return integrate.quad(special.erfcx, x, y)[0]


def ricciardi_exact(mu, sigma=0.01, tau=0.02, tau_rp=0.002, V_r=0.01, theta=0.02):
    min_u = (V_r - mu) / sigma
    max_u = (theta - mu) / sigma
    return 1.0 / (tau_rp + tau * torch.pi**0.5 * ierfcx_exact(-max_u, -min_u))


@pytest.mark.parametrize(
    "x",
    [
        torch.linspace(-0.01, 0.1, 1001),
        torch.linspace(-10.0, 50.0, 1001),
    ],
)
@pytest.mark.parametrize(
    "dtype", [torch.long, torch.bfloat16, torch.half, torch.float, torch.double]
)
def test_ricciardi(x, params, dtype):
    x = x.to(dtype)
    out = ricciardi(x, **params)
    expected = ricciardi_exact(x.double().numpy(), **params)
    expected = torch.from_numpy(expected).nan_to_num()
    if torch.is_floating_point(x):
        expected = expected.to(dtype)
    else:
        expected = expected.float()
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    "x",
    [
        torch.linspace(-0.01, 0.1, 1001, requires_grad=True).double(),
        torch.linspace(-10.0, 50.0, 1001, requires_grad=True).double(),
    ],
)
def test_ricciardi_grad(x, params):
    torch.autograd.gradcheck(functools.partial(ricciardi, **params), x)
