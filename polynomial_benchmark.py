import timeit
import argparse

import torch


def poly1d(coef, x):
    power = torch.arange(len(coef), device=x.device)
    return (coef * x[:, None]**power).sum(dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type=str, default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    
    c = torch.tensor([1.0, -0.5, 0.25, -0.125, 0.25, 0.2, -0.1, 0.05], device=device)
    x = torch.linspace(-5.0, 5.0, 100001, device=device)

    num, timing = timeit.Timer('poly1d(c, x)', globals={**locals(), **globals()}).autorange()
    print(f'Parallel: {timing / num:.5e}s')
    num, timing = timeit.Timer('c[0] + (c[1] + (c[2] + (c[3] + (c[4] + (c[5] + (c[6] + c[7] * x) * x) * x) * x) * x) * x) * x', globals={**locals(), **globals()}).autorange()
    print(f'Sequential: {timing / num:.5e}s')


if __name__ == '__main__':
    main()