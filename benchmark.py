import timeit
import argparse
import statistics

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
    parser.add_argument('--repeat', '-r', type=int, default=7)
    args = parser.parse_args()
    device = torch.device(args.device)

    for requires_grad in [False, True]:
        print(f'{requires_grad=}')
        x = torch.linspace(-10.0, 10.0, args.N, requires_grad=requires_grad, device=device)
        for name, func in {'dawsn': dawsn, 'erfi': erfi, 'ierfcx': ierfcx}.items():
            timings = timeit.Timer('func(x)', globals={**locals(), **globals()}).repeat(repeat=args.repeat, number=1)
            print(f'{name}: {format_time(statistics.median(timings))}')
    
        x = torch.linspace(-0.08, 1.0, args.N, requires_grad=requires_grad, device=device)
        timings = timeit.Timer('ricciardi(x)', globals={**locals(), **globals()}).repeat(repeat=args.repeat, number=1)
        print(f'ricciardi: {format_time(statistics.median(timings))}\n')

    print("backward pass")
    for name, func in {'dawsn': dawsn, 'erfi': erfi, 'ierfcx': ierfcx}.items():
        timings = timeit.Timer('y.backward()', setup='x = torch.linspace(-10.0, 10.0, args.N, requires_grad=True, device=device); y = func(x).sum()', globals={**locals(), **globals()}).repeat(repeat=args.repeat, number=1)
        print(f'{name}: {format_time(statistics.median(timings))}')
    timings = timeit.Timer('y.backward()', setup='x = torch.linspace(-0.08, 1.0, args.N, requires_grad=True, device=device); y = ricciardi(x).sum()', globals={**locals(), **globals()}).repeat(repeat=args.repeat, number=1)
    print(f'ricciardi: {format_time(statistics.median(timings))}')


if __name__ == '__main__':
    main()
