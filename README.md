#  BQNet 
BQNet.jl is a Julia package for distributional regression using Bernstein quantile networks (BQN). That is,
* the conditional distribution is specified by its quantile function and assumed to be a [Bernstein polynomial](https://en.wikipedia.org/wiki/Bernstein_polynomial) of a certain degree
* its distribution parameters, the coefficients of the Bernstein polynomial, are linked to the input feature variables by a neural network
* the model parameters, the weights and biases of the network, are estimated by optimising a composite quantile loss function.

The package is based on [Flux](https://fluxml.ai/).


##  Installation
BQNet can be installed by
```julia
using Pkg
Pkg.add(url = "https://github.com/jbbremnes/BQNet.jl")
```
or by entering REPL's package environment by pressing `]` and then
```julia
add "https://github.com/jbbremnes/BQNet.jl"
```

##  Examples
In the following examples the BQN is applied to various data
* [Synthetic data from conditional Gaussian mixtures](./examples/mixture_normal.md)
* ...

##  Background
... 


##  References
* [Bremnes, J. B. (2020)](https://doi.org/10.1175/MWR-D-19-0227.1). Ensemble Postprocessing Using Quantile Function Regression Based on Neural Networks and Bernstein Polynomials. Monthly Weather Review 148, 1, 403-414. [R code](https://github.com/jbbremnes/bqn_mwr).
* [Schulz, B. and Lerch, S. (2022)](https://doi.org/10.1175/MWR-D-21-0150.1). Machine learning methods for postprocessing ensemble forecasts of wind gusts: A systematic comparison. Monthly Weather Review, 150 (1), 235-257. doi:10.1175/MWR-D-21-0150.1
* [Gneiting, T., Lerch, S. and Schulz, B. (2023)](https://doi.org/10.1016/j.solener.2022.12.054). Probabilistic solar forecasting: Benchmarks, post-processing, verification. Solar Energy, 252 (1), 72-80. doi:10.1016/j.solener.2022.12.054
* [Höhlein, K., B. Schulz, R. Westermann, and S. Lerch (2024)](https://doi.org/10.1175/AIES-D-23-0070.1). Postprocessing of Ensemble Weather Forecasts Using Permutation-Invariant Neural Networks. Artif. Intell. Earth Syst., 3, doi:10.1175/AIES-D-23-0070.1.
