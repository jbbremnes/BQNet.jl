using Test
using BQNet, Statistics


@testset "Quantile loss" begin

    # basic
    qt = [[1,2] [2,3] [2,3]]
    y = [1, 2, 3]
    prob = [0.2, 0.5]
    @test qtloss(qt, y, prob) ≈ 0.2
    
    # one sample
    qt = [[1,2];; ]
    y = [1]
    prob = [0.2, 0.5]
    @test qtloss(qt, y, prob) ≈ 0.25

    # single quantile, two samples
    qt = [1 1]
    y = [1,2]
    prob = [0.5]
    @test qtloss(qt, y, prob) ≈ 0.25
    
    # equal quantiles and observations 
    qt = [1 1]
    y = [1,1]
    prob = [0.9]
    @test qtloss(qt, y, prob) ≈ 0.0

    # no aggregation
    @test qtloss(qt, y, prob; agg = identity) ≈ [0, 0]
    
end;

