#  useful layers and activation functions


#  softplus activation for BQN models
softplus_bqn(x::Matrix) = vcat(x[1:1, :], softplus(x[2:end, :]))



