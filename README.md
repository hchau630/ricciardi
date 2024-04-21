# About
An efficient, GPU-friendly, and differentiable PyTorch implementation of the Ricciardi transfer function based on equations and default parameters from [Sanzeni et al. (2020)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008165).

# Usage
For using the ricciardi function in your own code, you can either just copy the source file at `src/ricciardi/ricciardi.py` to your own code, or install the package in your python environment with `pip install ricciardi` and import the function with `from ricciardi import ricciardi`. To run tests, clone the repository, create a new environment, install the neccessary packages with `pip install -r requirements`, and run the command `pytest`.

# Benchmark
Compare performance with an interpolation-based approach. Forward pass is slightly slower, but backward pass is ~2x faster on GPU.

Results on CPU (AMD EPYC 7662, 8 cores) (`python benchmark/benchmark.py -N 100000 -r 100`):
```
forward pass, requires_grad=False
ricciardi: median=1.92 ms, min=1.88 ms (100 repeats)
ricciardi_interp: median=1.76 ms, min=1.72 ms (100 repeats)

forward pass, requires_grad=True
ricciardi: median=1.97 ms, min=1.93 ms (100 repeats)
ricciardi_interp: median=1.91 ms, min=1.8 ms (100 repeats)

backward pass
ricciardi: median=878 μs, min=842 μs (100 repeats)
ricciardi_interp: median=1.11 ms, min=1.08 ms (100 repeats)
```

Results on GPU (Nvidia A40) (`python benchmark/benchmark.py -N 100000 -r 100 --device cuda`):
```
forward pass, requires_grad=False
ricciardi: median=607 μs, min=596 μs (100 repeats)
ricciardi_interp: median=457 μs, min=448 μs (100 repeats)

forward pass, requires_grad=True
ricciardi: median=641 μs, min=628 μs (100 repeats)
ricciardi_interp: median=525 μs, min=515 μs (100 repeats)

backward pass
ricciardi: median=594 μs, min=575 μs (100 repeats)
ricciardi_interp: median=1.28 ms, min=1.25 ms (100 repeats)
```
