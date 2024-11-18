using Test
using BQNet


@testset "BQN model" begin

    x = rand(Float32, 10, 100)
    y = randn(Float32, 100) .+ x[1,:]
    tr_loader = Flux.DataLoader((x, y), batchsize = 10)
    degree = 9
    model = Chain(Dense(size(x,1), 32, elu),
                  Dense(32, degree + 1), softplus_bqn)
    bqnmodel = bqn_train!(model, tr_loader, tr_loader)
    @test bqnmodel isa BQNmodel
    @test all(bqnmodel.training_loss .> 0)
    @test all(bqnmodel.validation_loss .> 0)
    @test isnan(bqnmodel.censored_left)
    @test bqnmodel.degree == degree
    
end
