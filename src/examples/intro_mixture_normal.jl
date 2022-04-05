#  BQNet example based on data generated from a mixture normal model
#
#     Normal(1-2x, 0.5) + Normal(-1+2x, 0.25),  where X ~ Uniform(0, 1)
#
#  and equal weights of the components are assumed. 
#  input variables: p ordered samples for each x
#  target variable: one sample for each x
#

using BQNet
using Flux
using Distributions
using Plots


#  mixture model of two normal distributions
mixture(m; sd=0.25) =  MixtureModel([Normal(m, sd), Normal(-m, sd)])

#  generate data from the mixture normal model
#  n = #samples, p = #ordered samples as covariates
function create_data(; n = 10_000, p = 10, sd = 0.25)
    m   = -1 .+ 2 .* rand(n)
    x   = hcat([sort(rand(mixture(m[i]; sd=sd), p)) for i in eachindex(m)]...) 
    y   = hcat([rand(mixture(m[i]; sd=sd), 1) for i in eachindex(m)]...)[:]     
    return Float32.(x), Float32.(y)
end
  

#  create data and data loaders
n    = 50_000
p    = 10
x, y = create_data(n=n, p=p)
y    = y .+ 5f0
ktr  = rand(Bernoulli(0.9), n)
train_loader = Flux.Data.DataLoader((x[:, ktr], y[ktr]), batchsize = 500, shuffle = true, partial = false)
val_loader   = Flux.Data.DataLoader((x[:, .!ktr], y[.!ktr]), batchsize = 1)

#  create model
device = cpu
degree = 12
model  = Chain(Dense(size(x,1), 32, elu),
               Dense(32, 32, elu),
               Dense(32, degree+1, softplus)) |> device

#  train BQNet
@time fit = bqn_train!(model, train_loader, val_loader,
                       increasing = true, device = cpu)

#  make predictions
prob_out = Float32.(0.05:0.1:0.95)
u  = -1:0.01:1 
xp = Float32.(hcat([quantile(mixture(u[i]), (1:p)/(p+1)) for i in eachindex(u)]...))
fit_qts = quantile(fit, xp; prob = prob_out)

#  plot conditional quantiles
Plots.scalefontsizes(0.7)
plot(u, xp', color = :black, linestyle = :dot,
     legend = :top, label = reshape(["true quantiles"; fill("", 9)], 1, 10),
     title = "10 quantiles from mixture normal",
     xlab = "means of mixture components = ±U(-1,1)", ylab = "quantile")
plot!(u, fit_qts' .- 5, color = :blue,
      label = reshape(["BQN quantiles"; fill("", 9)], 1, 10))

# plot conditional cdf
yout = 5 .+ (-2:0.01:2)
fit_cdf = cdf2(fit, xp, Float32.(yout))
heatmap(u, yout .- 5, fit_cdf, title = "Conditional cumulative density",
        xlab = "means of mixture components = ±U(-1,1)", ylab = "target")


# plot conditional pdf
yout = Float32.(5 .+ (-2:0.01:2))
fit_pdf = pdf(fit, xp, yout; prob = Float32.(0:0.01:1))
heatmap(u, yout .- 5, fit_pdf, title = "Conditional density")
anim = @animate for i in axes(distr,2)
    plot(yout, distr[:, i], fill = true, yaxis = nothing, title = "Conditional density")
end
gif(anim, "bqn_mixture_normal.gif", fps = 20)

