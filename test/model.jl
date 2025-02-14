using Test
using BQNet
using Flux, MLUtils


@testset "BQN model" begin

    n = 100
    x = rand(Float32, 10, n)
    y = randn(Float32, n) .+ x[1,:]
    tr_loader = DataLoader((x, y), batchsize = 50)
    degree = 9
    model = Chain(Dense(size(x,1), 32, elu),
                  Dense(32, degree + 1), softplus_bqn)
    bqnmodel = bqn_train!(model, tr_loader, tr_loader)
    @test bqnmodel isa BQNmodel
    @test all(bqnmodel.training_loss .> 0)
    @test all(bqnmodel.validation_loss .> 0)
    @test isnan(bqnmodel.censored_left)
    @test bqnmodel.degree == degree

    pr = predict(bqnmodel, x; prob = Float32.(0:0.05:1))
    @test pr isa Matrix
    @test size(pr) == (21, n)

    cprob = cdf(bqnmodel, x, -1:0.1:3)
    @test cprob isa Matrix
    @test size(cprob) == (n, 41)
    
end;


