import timeit
import argparse
import statistics

import torch

from ricciardi import ricciardi
from ricciardi_interp import Ricciardi


def format_time(t):
    units = ["s", "ms", "μs", "ns"]
    i = 0
    while t < 1.0:
        t = t * 1.0e3
        i = i + 1
    return f"{t:.3g} {units[i]}"


def summary(timings, stats=["median", "min"]):
    funcs = {
        "median": statistics.median,
        "min": min,
        "max": max,
        "mean": statistics.mean,
        "stdev": statistics.stdev,
    }
    return (
        ", ".join(f"{k}={format_time(funcs[k](timings))}" for k in stats)
        + f" ({len(timings)} repeats)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", type=str, default="cpu")
    parser.add_argument("-N", type=int, default="10001")
    parser.add_argument("--repeat", "-r", type=int, default=7)
    parser.add_argument(
        "--stats",
        type=str,
        nargs="+",
        default=["median", "min"],
        choices=["median", "min", "max", "mean", "stdev"],
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    ricciardi_interp = Ricciardi().to(device)

    for requires_grad in [False, True]:
        print(f"forward pass, {requires_grad=}")
        x = torch.linspace(
            -0.08, 1.0, args.N, requires_grad=requires_grad, device=device
        )
        timings = timeit.Timer(
            "ricciardi(x)", globals={**locals(), **globals()}
        ).repeat(repeat=args.repeat, number=1)
        print(f"ricciardi: {summary(timings, args.stats)}")

        timings = timeit.Timer(
            "ricciardi_interp(x)", globals={**locals(), **globals()}
        ).repeat(repeat=args.repeat, number=1)
        print(f"ricciardi_interp: {summary(timings, args.stats)}\n")

    print("backward pass")
    timings = timeit.Timer(
        "y.backward()",
        setup="x = torch.linspace(-0.08, 1.0, args.N, requires_grad=True, device=device); y = ricciardi(x).sum()",
        globals={**locals(), **globals()},
    ).repeat(repeat=args.repeat, number=1)
    print(f"ricciardi: {summary(timings, args.stats)}")

    timings = timeit.Timer(
        "y.backward()",
        setup="x = torch.linspace(-0.08, 1.0, args.N, requires_grad=True, device=device); y = ricciardi_interp(x).sum()",
        globals={**locals(), **globals()},
    ).repeat(repeat=args.repeat, number=1)
    print(f"ricciardi_interp: {summary(timings, args.stats)}")


if __name__ == "__main__":
    main()
