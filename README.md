# BQNet 
BQNet is a Julia package for flexible distributional regression using Bernstein quantile networks (BQN). That is,
     * the conditional distribution is specified by its quantile function and assumed to be a [Bernstein polynomial](https://en.wikipedia.org/wiki/Bernstein_polynomial)
     * the distribution parameters (the Bernstein coefficients) are linked to the input variables/covariates by a neural network
     * the model parameters, the weights and biases of the network, are estimated by optimising a composite quantile loss function.

The package is based on [Flux](https://fluxml.ai/).

## Installation
BQNet can be installed by
"""julia
using Pkg
Pkg.add("https://github.com/jbbremnes/BQNet.jl")
"""
or by entering REPL's package environment by pressing `]` and then
"""julia
add "https://github.com/jbbremnes/BQNet.jl"
"""

## References
   * [Bremnes, J. B. (2020)](https://doi.org/10.1175/MWR-D-19-0227.1). Ensemble Postprocessing Using Quantile Function Regression Based on Neural Networks and Bernstein Polynomials. Monthly Weather Review 148, 1, 403-414.
   
