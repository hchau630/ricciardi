import torch
import numpy as np

sqrtpi = torch.pi**0.5


def dawsn(x):
    """
    Rational approximation of the Dawson function based on
    https://www.jstor.org/stable/2004886?seq=4
    """
    absx = x.abs()
    t = x**2
    return torch.where(
        absx < 3.5,
        torch.where(
            absx < 2.5,
            x * (1.08326558873e4 + (-1.28405832279e3 + (4.19672902280e2 + (-1.51982152422e1 + (1.67795116189e0 - 2.38594565696e-2 * t) * t) * t) * t) * t) / (1.08326558772e4 + (5.93771276935e3 + (1.48943557242e3 + (2.19728331833e2 + (1.99422336364e1 + t) * t) * t) * t) * t),
            (5.00652754437e-1 + 2.14221965778e-1 / (-4.91605365741e0 + t + 3.73902763512e1 / (4.07068101667e0 + t + 1.38216341182e1 / (-1.26817901598e1 + t + 8.87619386764e1 / (3.47393742586e0 + t))))) / x,
        ),
        torch.where(
            absx < 5.0,
            (5.0000965199e-1 + 2.4924632421e-1 / (-1.5853935006e0 + t - 5.3714272981e-1 / (-1.0385024821e1 + t + 1.7761778077e1 / (-4.1083233770e0 + t)))) / x,
            0.5 * (1 + (5.000381123e-1 + 7.449387745e-1 / (-2.748627766e0 + t)) / t) / x,
        ),
    )


# def dawsn2(x, N=6, h=0.4):
#     """
#     Dawson function evaluation based on Numerical Recipes, seems slower than rational approximation
#     """
#     A1, A2, A3 = 2.0 / 3.0, 0.4, 2.0 / 7.0
#     absx = x.abs()
#     x2 = x**2
#     n0 = 2 * (0.5 * absx / h + 0.5).round()
#     xp = absx - n0 * h
#     e1 = torch.exp(2 * xp * h)
#     e2 = e1**2
#     d1 = n0 + 1
#     d2 = d1 - 2
#     c = np.exp(-(np.arange(1, 2 * N, 2) * h)**2)

#     out = 0.0
#     for i in range(N):
#         out += c[i] * (e1 / d1 + 1 / (d2 * e1))
#         d1 += 2
#         d2 -= 2
#         e1 *= e2

#     return torch.where(
#         absx < 0.2,
#         x * (1.0 - A1 * x2 * (1.0 - A2 * x2 * (1.0 - A3 * x2))),
#         1 / sqrtpi * torch.exp(-xp**2) * x.sign() * out,
#     )


class Erfi(torch.autograd.Function):
    @staticmethod
    def forward(x):
        grad = 2 / sqrtpi * torch.exp(x**2)
        return (grad * dawsn(x), grad)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, grad = output
        ctx.save_for_backward(grad)

    @staticmethod
    def backward(ctx, grad_output, _):
        grad, = ctx.saved_tensors
        return grad * grad_output


def erfi(x):
    """
    Imaginary error function. Manual implementation of backward
    yields ~10x speedup on backward pass.
    """
    result, _ = Erfi.apply(x)
    return result


class Ierfcx(torch.autograd.Function):
    @staticmethod
    def forward(x):
        ax = x.abs()
        u = ax**-2
        v = (ax - 3.75) / (ax + 3.75)
        out = torch.where(
            ax >= 4.1,
            1 / sqrtpi * (torch.log(ax) + (0.25 + (-0.1875 + (0.3125 + (-0.8203125 + (2.953125 + (-13.53515625 + 75.41015625 * u) * u) * u) * u) * u) * u) * u) + 5.538959341195e-01,  # asymptotic expansion
            torch.log(1.0 + ax) * (8.402081835053e-01 + (-1.506407525002e-01 + (1.611762752505e-02 + (-1.336764713619e-02 + (-1.433259084023e-02 + (-1.050071231432e-02 + (-2.547932936248e-02 + (-1.672446952458e-02 - 7.747389892958e-03 * v) * v) * v) * v) * v) * v) * v) * v),  # chebyshev approximation
        )
        out = torch.where(x >= 0, out, out - 2 * torch.exp(ax**2) * dawsn(ax))
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return torch.special.erfcx(x) * grad_output


def ierfcx(x):
    """
    Integral of erfcx(t) from 0 to x. Manual implementation of backward
    yields ~15x speedup on backward pass.
    """
    return Ierfcx.apply(x)


def ricciardi(mu, tau=0.02, tau_rp=0.002, sigma_t=0.01, V_r=0.01, theta=0.02):
    dtype = mu.dtype
    mu = mu.double()  # need double precision input to get single precision output
    min_u = (V_r - mu) / sigma_t
    max_u = (theta - mu) / sigma_t

    if (-min_u).min() > -10:
        out = 1.0 / (tau_rp + tau * sqrtpi * (ierfcx(-min_u) - ierfcx(-max_u)))  # slightly faster path when there is no extreme value
    else:
        mask = -min_u > -10
        out = torch.empty_like(mu)
        out[mask] = 1.0 / (tau_rp + tau * sqrtpi * (ierfcx(-min_u[mask]) - ierfcx(-max_u[mask])))
        u = max_u[~mask]
        out[~mask] = u * torch.exp(-u**2) / (tau * sqrtpi)  # avoid NaNs - can't use torch.where due to NaN gradient
    
    return out.to(dtype)
