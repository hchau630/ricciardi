import itertools

import torch

from ricciardi import ricciardi


class Ricciardi(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # step-size of 5.0e-5 between -2 and 10
        points = torch.tensor([-1.0e4, *torch.linspace(-2.0, 10.0, 240001), 1.0e4])
        values = ricciardi(points, **kwargs)
        self._interpolator = RegularGridInterpolator((points,), values)

    def forward(self, x):
        return self._interpolator(x[None, ...])


# Code adapted from https://github.com/sbarratt/torch_interpolations
class RegularGridInterpolator(torch.nn.Module):
    def __init__(self, points, values):
        if not isinstance(points, (tuple, list)) or not all(
            isinstance(p, torch.Tensor) and p.ndim == 1 for p in points
        ):
            raise TypeError(
                "points must be a tuple or list of 1-dimensional torch.Tensor."
            )

        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor.")

        if tuple(len(p) for p in points) != values.shape:
            raise ValueError("shape of values is incompatible with points.")

        super().__init__()

        self.points = torch.nn.ParameterList(
            [torch.nn.Parameter(p, requires_grad=False) for p in points]
        )
        self.values = torch.nn.Parameter(values, requires_grad=False)
        self.ndim = len(points)

    def forward(self, x):
        if len(x) != self.ndim:
            raise ValueError(
                f"length of x, {len(x)}, must be equal to the number of grid dimensions, {self.ndim}."
            )

        idxs = []
        dists = []
        overalls = []
        for p, xi in zip(self.points, x):
            idx_right = torch.bucketize(xi, p)
            idx_right[idx_right >= p.shape[0]] = p.shape[0] - 1
            idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
            dist_left = xi - p[idx_left]
            dist_right = p[idx_right] - xi
            dist_left[dist_left < 0] = 0.0
            dist_right[dist_right < 0] = 0.0
            both_zero = (dist_left == 0) & (dist_right == 0)
            dist_left[both_zero] = dist_right[both_zero] = 1.0

            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        numerator = 0.0
        for indexer in itertools.product([0, 1], repeat=self.ndim):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
            numerator += self.values[as_s] * torch.stack(bs_s).prod(dim=0)
        denominator = torch.stack(overalls).prod(dim=0)
        return numerator / denominator
