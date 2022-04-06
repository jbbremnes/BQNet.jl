#  Bernstein quantile distribution networks as in Bremnes (2020).
#  To do: deal with 4d/5d data
#

using Printf: @sprintf
using LinearAlgebra: tril
using Interpolations: LinearInterpolation, ConstantInterpolation




"""
    BQNetModel

Struct for information related to the training and prediction of BQNet models.

# Elements
- `model`: Flux model
- `degree::UInt`: the degree of the Bernstein polynomials
- `increasing::Bool`: if true, the output of the `model` is treated as increments
- `training_prob::AbstractVector`: the quantile levels applied in the loss function during training
- `training_loss::AbstractVector`: the quantile loss on the training data for each epoch
- `validation_loss::AbstractVector`: the quantile loss on the validation data for each epoch
- `learning_rates::AbstractVector`: the learning rate for each epoch
- `epochs::UInt`: the total number of training epochs
"""
struct BQNetModel
    model
    degree::UInt
    increasing::Bool
    training_prob::AbstractVector
    training_loss::AbstractVector
    validation_loss::AbstractVector
    learning_rates::AbstractVector
    epochs::UInt
end

import Base.show
function show(io::IO, fit::BQNetModel)
    println(io, "Degree of Bernstein basis polynomials: ", fit.degree)
    println(io, "Non-decreasing Bernstein basis: ", fit.increasing)
    println(io, "Network: ", fit.model)
    println(io, "Network/model parameters: ", sum(length, Flux.params(fit.model)))
    trloss = findmin(fit.training_loss)
    println(io, "Best training loss: ", trloss[1], " (epoch ", trloss[2], 
            ", learning rate ", fit.learning_rates[trloss[2]], ")")
    valloss = findmin(fit.validation_loss)
    println(io, "Best validation loss: ", valloss[1], " (epoch ", valloss[2],
            ", learning rate ", fit.learning_rates[valloss[2]], ")")
    println(io, "Number of epochs: ", fit.epochs)
    println(io, "Initial learning rate: ", fit.learning_rates[1])
    println(io, "End learning rate: ", fit.learning_rates[fit.epochs])
end
    


timeit(tm::DateTime) = Dates.toms(Dates.now() - tm) / 1000
    

"""
    BernsteinMatrix(degree::UInt, prob::AbstractVector)

Compute the Bernstein basis polynomials of degree `degree` at `prob`. A matrix
of size (levels, degree+1) is returned.
"""
BernsteinMatrix(degree::Integer, prob::AbstractVector) =
    [binomial(degree, d) * p^d * (1-p)^(degree-d) for p in prob, d in 0:degree]
    
    

"""
    bqn_train!(model, tr_loader, val_loader;
               increasing::Bool = true,
               prob::AbstractVector = Float32.(0.01:0.07:0.99), 
               learning_rate::AbstractFloat = 0.001,
               learning_rate_scale::AbstractFloat = 0.1,
               learning_rate_min::AbstractFloat = 5e-6,
               patience::UInt = 10, max_epochs::UInt = 100,
               best_model::Bool = true, device::Function = cpu)

A customized training loop for Bernstein quantile distribution networks with early stopping.

# Arguments
- `model`: the Flux model. The size of the network output determines the degree of the
         Bernstein polynomial (degree = size(output) - 1). It is recommended not to
         set the degree too low. A degree between 8 and 16, say, is often a good choice.
- `tr_loader`: the data loader for the training data
- `val_loader`: the data loader for the validation data
- `increasing::Bool=true`: if true, the output from the `model` is assumed to be increments
         of subsequent Bernstein coefficients, that is, [α0, α1-α0, α2-α1, ...].
         The quantile function can then be forced to be non-decreasing by choosing an
         activation function in the output layer that restricts all increments
         apart from the first to be non-negative. Note that restricting the first
         Bernstein coefficient (α0) to be non-negative implies that the quantile functions
         also will be non-negative which may not be desireable. To avoid this, one option
         is to add a sufficiently large positive offset value to the target variable prior
         to training such that the positivity constraint of the first coefficient have no
         effect. If so, the offset value to any predicted quantile should be subtracted
         afterwards.
- `prob::AbstractVector=Float32.(0.01:0.07:0.99)`: quantile levels for the quantile loss
         function. 
- `learning_rate::AbstractFloat=0.001`: the initial learning rate
- `learning_rate_scale::AbstractFloat=0.1`: the scaling factor applied to change the
         learning rate
- `learning_rate_min::AbstractFloat=5e-6`:  the learning rate at which the training will
         be stopped
- `patience::Integer=10`: the number of epochs with no improvement of the validation score
         before the learning_rate is changed
- `max_epochs::Integer=100`: the maximum number of epochs
- `best_model::Bool=true`: if true return the best model. Otherwise the last model.
- `device::Function=cpu`: the compute device; either `cpu` or `gpu`.

Return a struct of type BQNetModel. Note that the variable `model` is updated and can be
used for further training. The training will stop if either `max_epochs` or
`learning_rate_min` is reached.

For the moment only vector output of the network (single univariate distribution)
is supported.

# Examples
See ...

# References
- Bremnes, J.B. (2020). Ensemble Postprocessing Using Quantile Function Regression Based
    on Neural Networks and Bernstein Polynomials, Monthly Weather Review, 148(1), 403-414.
    https://doi.org/10.1175/MWR-D-19-0227.1
"""
function bqn_train!(model, tr_loader, val_loader;
                    increasing::Bool = true,
                    prob::AbstractVector = Float32.(0.01:0.07:0.99), 
                    learning_rate::AbstractFloat = 0.001,
                    learning_rate_scale::AbstractFloat = 0.1,
                    learning_rate_min::AbstractFloat = 5e-6,
                    patience::Integer = 10, max_epochs::Integer = 100,
                    best_model::Bool = true, device::Function = cpu)

    prob_tr    = Float32.(prob)
    #degree     = size(Flux.params(model[end])[1], 1) - 1   
    degree     = size(model(first(tr_loader)[1]))[end-1] - 1
    B          = BernsteinMatrix(degree, prob_tr)
    if increasing
        B = B * tril(fill(1f0, (degree+1,degree+1))) 
    end
    B          = B |> device
    prob_tr    = prob_tr |> device
    loss(x, y) = qtloss(B * model(x), y, prob_tr)   
    prm        = Flux.params(model) 
    opt        = ADAM(learning_rate)
    qs_tr      = []  
    qs_val     = []  
    lrs        = []  
    bmodel     = deepcopy(model)
    ictr       = 1
    epochs     = 0
    
    for i in 1:max_epochs

        epochs += 1
        push!(lrs, opt.eta)
        tm_tr  = Dates.now()
        local trloss = 0f0
        for (xt, yt) in tr_loader
            xt  = xt |> device
            yt  = yt |> device
            gs = gradient(prm) do
                trloss += loss(xt, yt)
                return trloss
            end
            Flux.update!(opt, prm, gs)
        end
        push!(qs_tr, trloss / length(tr_loader))
        tm_tr     = timeit(tm_tr)
        
        tm_val  = Dates.now()
        valloss = zero(eltype(prob))
        n       = 0
        for (x, y) in val_loader
            x = x |> device
            y = y |> device
            valloss += loss(x, y) * size(x)[end]
            n       += size(x)[end]
        end
        push!(qs_val, valloss / n)
        tm_val    = timeit(tm_val)
        
        lr        = ""
        if findmin(qs_val)[2] == i   # last fit the best?
            if best_model
                bmodel = deepcopy(model)
            end
            ictr = 1
        else
            if ictr > patience
                opt.eta *= learning_rate_scale
                lr       = string(opt.eta)
                ictr     = 1
            else
                ictr    += 1
            end
        end
        @info @sprintf("%4d: quantile scores training and validation: %.5f  %.5f %s  %.3fs/%.3fs %s",
                       i, qs_tr[i], qs_val[i], i == findmin(qs_val)[2] ? "*" : " ",
                       tm_tr, tm_val,
                       lr == "" ? lr : string("\n      new learning rate: ", lr))
        if opt.eta < learning_rate_min
            break
        end
    end

    println("best quantile validation score: ", findmin(qs_val)) 
    if best_model
        model = deepcopy(bmodel)
    end

    return BQNetModel(cpu(model), degree, increasing, cpu(prob_tr),
                      qs_tr, qs_val, lrs, epochs)
end



import Statistics: quantile
"""
    quantile(fit::BQNetModel, x; prob::AbstractVector = fit.training_prob)

Compute conditional quantiles for levels `prob` at `x` based on the BQN model `fit`.
"""
function quantile(fit::BQNetModel, x;
                  prob::AbstractVector = fit.training_prob)
    B = BernsteinMatrix(fit.degree, Float32.(prob))
    if fit.increasing
        B *= tril(fill(1f0, (fit.degree+1, fit.degree+1))) 
    end
    return B * fit.model(x)
end


function aa(x)
    idx = Int64[]
    for i in 1:size(x,1)-1
        if x[i] != x[i+1]
            push!(idx, i)
        end
    end
    return idx
end


"""
    cdf(fit::BQNetModel, x, y::AbstractVector; prob::AbstractVector = Float32.(0:0.01:1)) 

Compute the conditional cumulative distribution function of Y|x for values `y` for each `x`
based on the BQN model `fit`. The CDFs evaluated at `y` are obtained by linear interpolation
of predicted quantiles at levels `prob`. Hence, the size of `prob` determines the accuracy of
the approximation.
"""
function cdf(fit::BQNetModel, x, y::AbstractVector;
             prob::AbstractVector = Float32.(0:0.01:1))
    p   = Float32.(prob)
    yy  = Float32.(y)
    qts = quantile(fit, x, prob = p)
    out = zeros(Float32, length(y), size(x)[end])
    @views for i in axes(qts)[end]
        try
            out[:, i] = LinearInterpolation(qts[:, i], p, extrapolation_bc = Flat())(yy)
        catch e
            println("error for sample ", i, ". Increasing quantiles: ",
                    sort(qts[:,i]) == qts[:,i])
        end
    end
    return out
end


"""
    quantile_density(fit::BQNetModel, x; prob::AbstractVector = Float32.(0:0.01:1))

Compute the conditional quantile density for levels `prob` for each `x` based on 
the BQN model `fit`. Quantile density is the derivative of the quantile function.
"""
function quantile_density(fit::BQNetModel, x;
                          prob::AbstractVector = Float32.(0:0.01:1))
    p  = Float32.(prob)
    B  = BernsteinMatrix(fit.degree-1, p)
    c0 = zeros(Float32, length(p))
    B  = fit.degree .* (hcat(c0, B) .- hcat(B, c0))
    if fit.increasing
        B *= tril(fill(1f0, (fit.degree+1, fit.degree+1))) 
    end
    return B * fit.model(x)
end


"""
    pdf(fit::BQNetModel, x, y; prob::AbstractVector = Float32.(0:0.01.1))

Compute the conditional probability density function of Y|x at values `y` for each `x`
based on the BQN model `fit`. Note that the computation relies on linear interpolation
and its accuracy depends on the size of `prob`. `prob` should preferably include 0 and 1.
"""
function pdf(fit::BQNetModel, x, y::AbstractVector;
             prob::AbstractVector = Float32.(0:0.01:1))
    p     = Float32.(prob)
    q     = quantile(fit, x; prob = p)
    DqInv = 1f0 ./ quantile_density(fit, x, prob = p)
    out   = zeros(eltype(y), length(y), size(x)[end])
    @views for i in axes(DqInv)[end]
        try
            pout      = LinearInterpolation(q[:, i], p, extrapolation_bc = Flat())(y)
            out[:, i] = LinearInterpolation(p, DqInv[:, i], extrapolation_bc = Flat())(pout)
            k         = (y .> maximum(q[:,i])) .| (y .< minimum(q[:,i]))
            out[k,i] .= 0f0  # set pdf for ys beyond extrema(q[:,i]) to zero
        catch e
            println("error for sample ", i)
        end
    end
    return out
end



  
