##  Conditional Gaussian mixtures with a point mass
This example is similar to the [mixture normal example](./mixture_normal.md) except that a point mass at -0.5 is introduced by truncating all values below. One way to deal with this is to treat -0.5 as left censored values. Here is the code.

```julia
using BQNet, Flux, Random, Distributions, Plots

#  define mixture model
mixture(m, sd) = MixtureModel([Normal(m, sd), Normal(-m, sd)])

#  function for generating data
function create_data(; n = 50_000, sd = 0.25, cens = -0.5)
    x = Float32.(rand(Uniform(-1, 1), n))
    y = map(m -> rand(mixture(m,sd), 1)[1], x)
    y[y .< cens] .= cens
    return reshape(x, 1, :), Float32.(y)
end

#  create data
Random.seed!(1234)
n = 20_000
x, y = create_data(n = n)

#  create data loaders for training and validation
idx = Int(round(0.9 * n))
ktr = 1:idx
kval = idx+1:n
train_loader = Flux.DataLoader((x[:, ktr], y[ktr]), batchsize = 250,
                               shuffle = true, partial = false)
val_loader   = Flux.DataLoader((x[:, kval], y[kval]), batchsize = length(kval))


#  create and train model
device = cpu
degree = 12
model  = Chain(Dense(size(x,1), 32, elu),
               Dense(32, 32, elu),
               Dense(32, degree+1), softplus_bqn) |> device

@time model_fit = bqn_train!(model, train_loader, val_loader;
                             censored_left = -0.5f0, use_censored_prob = true,
                             device = device)


#  make predictions of quantiles 
prob_out = 0.05:0.1:0.95
xp = Float32.(transpose(-1:0.01:1))
qts_true = [quantile(mixture(x, 0.25), prob_out) for x in xp[:]]
qts_true = Float32.(reduce(hcat, qts_true))
qts_true[qts_true .< -0.5] .= -0.5
fit_qts  = predict(model_fit, xp; prob = prob_out)


#  plot true and fitted distributions
plot(u, qts_true', color = :black, linestyle = :dot,
     legend = :top, label = reshape(["true quantiles"; fill("", 9)], 1, 10),
     title = "Prediction of $(length(prob_out)) quantiles from a censored mixture normal with point mass at -0.5",
     xlab = "means of mixture components (±)", ylab = "quantile")
plot!(u, fit_qts', color = :blue,
      label = reshape(["BQN quantiles"; fill("", 9)], 1, 10))
```
![BQN fit](./mixture_normal/censored_mixture_normal.png)
