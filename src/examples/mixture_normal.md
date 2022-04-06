## Mixture of normal distributions

In this example, synthetic data from a mixture of two normal distributions defined by
```
   Normal(M, 0.25^2) + Normal(-M, 0.25^2)  where M ~ Uniform(-1, 1) 
```   
are applied to train a BQN model. The data is generated in two steps
* `n = 50_000` samples from the `U(-1, 1)`
* for each of the `n` mixture distributions generate
  * `p = 10` ordered samples as input variables/covariates
  * one target sample

```julia
using BQNet, Flux, Distributions, Plots

#  define mixture models
mixture(m, sd) = MixtureModel([Normal(m, sd), Normal(-m, sd)])

#  function for generating data
function create_data(; n = 50_000, p = 10, sd = 0.25)
    m   = rand(Uniform(-1, 1), n)
    x   = hcat([sort(rand(mixture(m[i], sd), p)) for i in eachindex(m)]...) 
    y   = hcat([rand(mixture(m[i], sd), 1) for i in eachindex(m)]...)[:]     
    return Float32.(x), Float32.(y)
end

#  create data
n, p = 50_000, 10
x, y = create_data(n=n, p=p)
y    = y .+ 5f0                  # add an offset for constraining the quantile function

#  create data loaders
ktr  = rand(Bernoulli(0.9), n)   # ~90% training data, ~10% for validation
train_loader = Flux.Data.DataLoader((x[:, ktr], y[ktr]), batchsize = 500,
                                    shuffle = true, partial = false)
val_loader   = Flux.Data.DataLoader((x[:, .!ktr], y[.!ktr]), batchsize = 1)

#  create model
device = cpu
degree = 12
model  = Chain(Dense(size(x,1), 32, elu),
               Dense(32, 32, elu),
               Dense(32, degree+1, softplus)) |> device


#  train BQNet model
@time fit = bqn_train!(model, train_loader, val_loader,
                       increasing = true, device = cpu)


#  make predictions of quantiles and plot
prob_out = 0.05:0.1:0.95
u        = -1:0.01:1 
xp       = [quantile(mixture(u[i], 0.25), (1:p)/(p+1)) for i in eachindex(u)]
xp       = Float32.(hcat(xp...))
fit_qts  = quantile(fit, xp; prob = prob_out)

Plots.scalefontsizes(0.7)
plot(u, xp', color = :black, linestyle = :dot,
     legend = :top, label = reshape(["true quantiles"; fill("", 9)], 1, 10),
     title = "10 quantiles from mixture normal",
     xlab = "means of mixture components = ±U(-1,1)", ylab = "quantile")
plot!(u, fit_qts' .- 5, color = :blue,
      label = reshape(["BQN quantiles"; fill("", 9)], 1, 10))

#  plot conditional CDF
yout    = (-2:0.01:2) .+ 5
fit_cdf = cdf2(fit, xp, Float32.(yout))
heatmap(u, yout .- 5, fit_cdf, title = "Conditional cumulative density",
        xlab = "means of mixture components = ±U(-1,1)", ylab = "target")

# plot conditional pdf
yout    = Float32.(5 .+ (-2:0.01:2))
fit_pdf = pdf(fit, xp, yout; prob = 0:0.01:1)
heatmap(u, yout .- 5, fit_pdf, title = "Conditional density")
anim = @animate for i in axes(distr,2)
    plot(yout, fit_pdf[:, i], fill = true, yaxis = nothing, title = "Conditional density")
end
gif(anim, "bqn_mixture_normal.gif", fps = 20)

```
