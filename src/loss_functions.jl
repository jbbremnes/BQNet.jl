

"""
    qtloss(qt, y, prob; agg = mean)

Computes the composite quantile loss for the levels given by `prob`.    
```
 qt    quantiles of size (prob, samples)
 y     observations
 prob  quantile levels/probabilities
 agg   aggregation function with `mean` as default. Other useful functions are `identity`,
       `u -> mean(w .* u)` (weighted mean), ...
```
"""
function qtloss(qt::AbstractMatrix, y::AbstractVector, prob::AbstractVector; agg = mean)
    err = y .- qt'
    return agg( (prob' .- (err .< 0)) .* err )
end

function qtloss(qt::AbstractArray{T,4}, y::AbstractArray{T,4}, prob::AbstractVector{T}; agg = mean) where T<:Real
    err  = y .- qt
    prob = reshape(prob, 1, 1, :, 1)
    agg((prob .- (err .< 0)) .* err)
end

function qtloss(qt::AbstractArray{T,5}, y::AbstractArray{T,5}, prob::AbstractVector{T}; agg = mean) where T<:Real
    err  = y .- qt
    prob = reshape(prob, 1, 1, 1, :, 1)
    agg((prob .- (err .< 0)) .* err)
end




"""
    brier_score(prob, y; agg = mean)

Computes the Brier score
```
 prob  probability predictions
 y     observations true/false
```
"""
function brier_score(prob::AbstractVector, y::AbstractVector; agg = mean)
     return agg( (prob .- y).^2 )
end




# crps_normal, ....
