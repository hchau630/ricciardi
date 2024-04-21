# About
An efficient, GPU-friendly, and differentiable PyTorch implementation of the Ricciardi transfer function based on equations and default parameters from [Sanzeni et al. (2020)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008165).

# Usage
For using the ricciardi function in your own code, you can either just copy the source file at `src/ricciardi/ricciardi.py` to your own code, or install the package in your python environment with `pip install ricciardi` and import the function with `from ricciardi import ricciardi`. To run tests, clone the repository, create a new environment, install the neccessary packages with `pip install -r requirements`, and run the command `pytest`.

# Benchmark
Compare performance with an interpolation-based approach. Forward pass is slightly slower, but backward pass is >2x faster on GPU.

Results on CPU (AMD EPYC 7662, 8 cores) (`python benchmark/benchmark.py -N 100000 -r 100`):
```
forward pass, requires_grad=False
ricciardi: median=1.86 ms, min=1.84 ms (100 repeats)
ricciardi_interp: median=1.75 ms, min=1.72 ms (100 repeats)

forward pass, requires_grad=True
ricciardi: median=1.94 ms, min=1.9 ms (100 repeats)
ricciardi_interp: median=1.92 ms, min=1.75 ms (100 repeats)

backward pass
ricciardi: median=814 μs, min=796 μs (100 repeats)
ricciardi_interp: median=1.17 ms, min=1.15 ms (100 repeats)
```

Results on GPU (Nvidia A40) (`python benchmark/benchmark.py -N 100000 -r 100 --device cuda`):
```
forward pass, requires_grad=False
ricciardi: median=517 μs, min=508 μs (100 repeats)
ricciardi_interp: median=460 μs, min=453 μs (100 repeats)

forward pass, requires_grad=True
ricciardi: median=556 μs, min=549 μs (100 repeats)
ricciardi_interp: median=527 μs, min=520 μs (100 repeats)

backward pass
ricciardi: median=463 μs, min=364 μs (100 repeats)
ricciardi_interp: median=1.11 ms, min=1.09 ms (100 repeats)
```
