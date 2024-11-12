##  Conditional Gaussian mixture data

In this it is demonstrated how Bernstein Quantile Networks can be trained on data generated from a conditional mixture of two normal distributions. The generating model is defined as
```
    μ ~ Uniform(-1, 1)
    Normal(μ, 0.25^2)/2 + Normal(-μ, 0.25^2)/2 
```   

The synthetic data is generated in two steps
* 50_000 samples from the Uniform(-1, 1) to define the component means 
* for each mixture distribution
  * 10 samples are generated, ordered and assumed to be 10 input variables/covariates
  * 1 sample is generated as target variable


The Julia code is as follows

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
x, y = create_data(n = n, p = p)
y    = y .+ 5f0                  # add an offset for constraining the quantile function

#  create data loaders
ktr  = rand(Bernoulli(0.9), n)   # ~90% training data => ~10% for validation
kval = .!ktr
train_loader = Flux.Data.DataLoader((x[:, ktr], y[ktr]), batchsize = 500,
                                    shuffle = true, partial = false)
val_loader   = Flux.Data.DataLoader((x[:, kval], y[kval]), batchsize = length(kval))

#  create model
device = cpu
degree = 12
model  = Chain(Dense(size(x,1), 32, elu),
               Dense(32, 32, elu),
               Dense(32, degree+1, softplus)) |> device

#  train BQNet model
@time fit = bqn_train!(model, train_loader, val_loader, increasing = true, device = device)

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
fit_cdf = cdf(fit, xp, Float32.(yout))
heatmap(u, yout .- 5, fit_cdf, title = "Conditional cumulative density",
        xlab = "means of mixture components = ±U(-1,1)", ylab = "target")

# plot conditional PDF
yout    = (-2:0.01:2) .+ 5
fit_pdf = pdf(fit, xp, yout; prob = 0:0.01:1)
heatmap(u, yout .- 5, fit_pdf, title = "Conditional density")
anim = @animate for i in axes(distr,2)
    plot(yout, fit_pdf[:, i], fill = true, yaxis = nothing, title = "Conditional density")
end
gif(anim, "bqn_mixture_normal.gif", fps = 20)
```

Note that the BQN model has not been tuned. Hence, better fits should be possible by exploring other network configurations and hyper-parameters including the degree of the Bernstein polynomial.

