module BQNet

using Flux
using Dates
using Statistics
using LinearAlgebra


include("bqn.jl")
export BQNmodel, bqn_train!, quantile, predict, cdf

include("loss_functions.jl")
export qtloss

include("layers.jl")
export Ensemble

#include("activations.jl")
softplus_bqn(x) = vcat(x[1:1, :], softplus(x[2:end, :]))
export softplus_bqn


end # module
