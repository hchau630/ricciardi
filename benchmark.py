import timeit
import argparse

import torch

from ricciardi import dawsn, erfi, ierfcx, ricciardi


def format_time(t):
    units = ['s', 'ms', 'μs', 'ns']
    i = 0
    while t < 1.0:
        t = t * 1.0e3
        i = i + 1
    return f'{t:.3g} {units[i]}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type=str, default='cpu')
    parser.add_argument('-N', type=int, default='10001')
    args = parser.parse_args()
    device = torch.device(args.device)
    
    x = torch.linspace(-10.0, 10.0, args.N, device=device)

    for name, func in {'dawsn': dawsn, 'erfi': erfi, 'ierfcx': ierfcx}.items():
        num, timing = timeit.Timer('func(x)', globals={**locals(), **globals()}).autorange()
        print(f'{name}: {format_time(timing / num)}')

    x = torch.linspace(-0.08, 1.0, args.N, device=device)
    num, timing = timeit.Timer('ricciardi(x)', globals={**locals(), **globals()}).autorange()
    print(f'ricciardi: {format_time(timing / num)}')


if __name__ == '__main__':
    main()
