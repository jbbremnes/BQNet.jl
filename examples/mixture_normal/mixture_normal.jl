using BQNet, Flux, Distributions, Plots

#  define mixture model
mixture(m, sd) = MixtureModel([Normal(m, sd), Normal(-m, sd)])

#  function for generating data
function create_data(; n = 50_000, sd = 0.25)
    x = Float32.(rand(Uniform(-1, 1), n))
    y = map(m -> rand(mixture(m,sd), 1)[1], x)
    return reshape(x, 1, :), Float32.(y)
end

#  create data
n = 50_000
x, y = create_data(n = n)

#  create data loaders
idx = Int(round(0.9 * n))
ktr = 1:idx
kval = idx+1:n
train_loader = Flux.DataLoader((x[:, ktr], y[ktr]), batchsize = 500,
                                    shuffle = true, partial = false)
val_loader   = Flux.DataLoader((x[:, kval], y[kval]), batchsize = length(kval))


#  create and train model
device = cpu
degree = 24
model  = Chain(Dense(size(x,1), 32, elu),
               Dense(32, 32, elu),
               Dense(32, degree+1), softplus_bqn) |> device

@time model_fit = bqn_train!(model, train_loader, val_loader; device = device)


#  make predictions of quantiles 
prob_out = 0.05:0.1:0.95
xp = Float32.(transpose(-1:0.01:1))
qts_true = [quantile(mixture(x, 0.25), prob_out) for x in xp[:]]
qts_true = Float32.(reduce(hcat, qts_true))
fit_qts  = predict(model_fit, xp; prob = prob_out)


#  plot true and fitted distributions
plot(u, qts_true', color = :black, linestyle = :dot,
     legend = :top, label = reshape(["true quantiles"; fill("", 9)], 1, 10),
     title = "Prediction of $(length(prob_out)) quantiles from a mixture normal",
     xlab = "means of mixture components (±)", ylab = "quantile")
plot!(u, fit_qts', color = :blue,
      label = reshape(["BQN quantiles"; fill("", 9)], 1, 10))



