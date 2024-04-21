# Ricciardi
An efficient, GPU-friendly, and differentiable PyTorch implementation of the Ricciardi transfer function based on equations and default parameters from [Sanzeni et al. (2020)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008165).

# Usage
For using the ricciardi function in your own code, you can either just copy the source file at `src/ricciardi/ricciardi.py` to your own code, or install the package in your python environment with `pip install ricciardi` and import the function with `from ricciardi import ricciardi`. To run tests, clone the repository, create a new environment, install the neccessary packages with `pip install -r requirements`, and run the command `pytest`.

# Benchmark
Compare performance with an interpolation-based approach, by running the script `benchmark/benchmark.py`.

Results on CPU (M1 Macbook Pro):
```
forward pass, requires_grad=False
ricciardi: median=405 μs, min=384 μs (100 repeats)
ricciardi_interp: median=527 μs, min=432 μs (100 repeats)

forward pass, requires_grad=True
ricciardi: median=411 μs, min=390 μs (100 repeats)
ricciardi_interp: median=540 μs, min=462 μs (100 repeats)

backward pass
ricciardi: median=174 μs, min=172 μs (100 repeats)
ricciardi_interp: median=226 μs, min=204 μs (100 repeats)
```