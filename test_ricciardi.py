import functools

import numpy as np
from scipy import integrate, special
import torch
import pytest

from ricciardi import dawsn, erfi, ierfcx, ricciardi


@pytest.fixture
def params():
    return dict(tau=0.02, tau_rp=0.002, sigma_t=0.01, V_r=0.01, theta=0.02)


@np.vectorize
def ierfcx_exact(x):
    return integrate.quad(special.erfcx, 0, x)[0]


def ricciardi_exact(mu, tau=0.02, tau_rp=0.002, sigma_t=0.01, V_r=0.01, theta=0.02):
    min_u = (V_r - mu) / sigma_t
    max_u = (theta - mu) / sigma_t
    return 1.0 / (tau_rp + tau * np.sqrt(np.pi) * (ierfcx_exact(-min_u) - ierfcx_exact(-max_u)))


@pytest.mark.parametrize('x', [
    torch.linspace(-5.0, 5.0, 1001),
    torch.linspace(-10.0, 10.0, 1001).double(),
])
def test_erfi(x):
    out = erfi(x)
    expected = torch.from_numpy(special.erfi(x.numpy()))
    torch.testing.assert_close(out, expected)


def test_erfi_grad():
    x = torch.linspace(-10.0, 10.0, 1001, requires_grad=True).double()
    p = torch.tensor([-5.0, -3.5, -2,5, 2.5, 3.5, 5.0])  # numerical gradient is inaccurate at these points
    x = x[(x - p[:, None]).abs().min(dim=0).values > 1.0e-5]
    torch.autograd.gradcheck(erfi, x)


@pytest.mark.parametrize('x', [
    torch.linspace(-5.0, 5.0, 1001),
    torch.linspace(-10.0, 10.0, 1001).double(),
])
def test_ierfcx(x):
    out = ierfcx(x)
    expected = torch.from_numpy(ierfcx_exact(x.double().numpy())).to(out.dtype)
    torch.testing.assert_close(out, expected)


def test_ierfcx_grad():
    x = torch.linspace(-10.0, 50.0, 1001, requires_grad=True).double()
    p = torch.tensor([-5.0, -3.5, -2,5, 4.1])  # numerical gradient is inaccurate at these points
    x = x[(x - p[:, None]).abs().min(dim=0).values > 1.0e-5]
    torch.autograd.gradcheck(ierfcx, x)


@pytest.mark.parametrize('x', [
    torch.linspace(-0.01, 0.1, 1001),
    torch.linspace(-10.0, 100.0, 1001),
])
def test_ricciardi(x, params):
    out = ricciardi(x, **params)
    expected = ricciardi_exact(x.double().numpy(), **params)
    expected = torch.from_numpy(expected).float().nan_to_num()
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize('x', [
    torch.linspace(-0.01, 0.1, 1001, requires_grad=True).double(),
    torch.linspace(-10.0, 100.0, 1001, requires_grad=True).double(),
])
def test_ricciardi_grad(x, params):
    p = torch.tensor([-5.0, -3.5, -2,5, 4.1])  # numerical gradient is inaccurate at these points
    p = torch.cat([p * params['sigma_t'] + params['V_r'], p * params['sigma_t'] + params['theta']])
    x = x[(x - p[:, None]).abs().min(dim=0).values > 1.0e-5]
    torch.autograd.gradcheck(functools.partial(ricciardi, **params), x, rtol=0.005)
    