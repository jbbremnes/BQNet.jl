#  Useful layers:
#     - Ensemble      for ensemble input of size (features, members, batch)
#


#  softplus activation for BQN models
#softplus_bqn(x::Matrix) = vcat(x[1:1, :], softplus(x[2:end, :]))
softplus_bqn(x) = vcat(x[1:1, :], softplus(x[2:end, :]))


#  layer for ensemble data of size (features, members, batch)
#  a shared network is applied separately to each member.
struct Ensemble{T <: Chain}
    chain::T
end

function (ens::Ensemble)(x)
    x2 = reshape(x, size(x,1), :)
    z  = ens.chain(x2)
    return reshape(z, size(z,1)*size(x,2), size(x,3))
end

Flux.@layer Ensemble

#  put this into test or examples
if false
    x = rand(Float32, 4, 10, 100)
    y = rand(Float32, 2, 100)
    model = Chain(Ensemble(Chain(Dense(4 => 1, elu))),
                  Dense(10 => 2, elu))
    loader = Flux.DataLoader((x, y), batchsize = 50, shuffle = true)
    optim = Flux.setup(Flux.Adam(0.01), model)
    for (x, y) in loader
        loss, grads = Flux.withgradient(model) do m
            Flux.Losses.mse(m(x), y)
        end
        Flux.update!(optim, model, grads[1])
    end  
end
