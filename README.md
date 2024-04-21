# Ricciardi
An efficient, GPU-friendly, and differentiable PyTorch implementation of the Ricciardi transfer function based on equations and default parameters from [Sanzeni et al. (2020)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008165).

# Usage
For using the ricciardi function in your own code, you can either just copy the source file at `src/ricciardi/ricciardi.py` to your own code, or install the package in your python environment with `pip install ricciardi` and import the function with `from ricciardi import ricciardi`. To run tests, clone the repository, create a new environment, install the neccessary packages with `pip install -r requirements`, and run the command `pytest`.
