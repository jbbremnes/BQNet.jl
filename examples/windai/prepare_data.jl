#   read and prepare data
#
#

using DataFrames, Dates
using CSV, JLD2


#  read MEPS data from CSV file and convert to 4d array
function read_meps(; fname = "./data/meps_statnett.csv")

    # read csv file
    df = CSV.read(fname, DataFrame)   # ~3 minutes, ~6.8M rows
    @views for j in 5:ncol(df)
        df[!, j] .= Float32.(df[!, j])
    end
    
    # extract and sort dimension values
    sites = sort(unique(df.sid))
    time_refs = sort(unique(df.time_ref))
    lts = sort(unique(df.lt))
    vars = ["ws10m", "wd10m", "t2m", "rh2m", "mslp", "g10m"]
    mbrs = lpad.(string.(0:14), 2, '0')

    varmbr  = ["$(v)_$(m)" for m in mbrs, v in vars][:]
    @assert varmbr == names(df)[5:end]
    
    # build index maps
    site_idx = Dict(s => i for (i, s) in enumerate(sites))
    time_ref_idx = Dict(t => i for (i, t) in enumerate(time_refs))
    lt_idx = Dict(l => i for (i, l) in enumerate(lts))
 
    # allocate array
    data = Array{Float32}(undef, length(sites), length(mbrs), length(vars)+1, length(lts), length(time_refs))
    
    # fill array
    @views Threads.@threads for i in 1:nrow(df)
        s = site_idx[df.sid[i]]
        tr = time_ref_idx[df.time_ref[i]]
        l = lt_idx[df.lt[i]]
        data[s, :, 1:end-1, l, tr] .= reshape(collect(df[i,5:end]), length(mbrs), length(vars))
        data[s, :, end, l, tr] .= df.lt[i]
    end  
    push!(vars, "leadtime")

    #  reduce to 4d array here!
    data = reshape(data, size(data,1), size(data,2), size(data,3), :)
    tm = vec(Hour.(lts) .+ permutedims(time_refs))
    lt = repeat(lts, outer = length(time_refs))
    tmr = repeat(time_refs, inner = length(lts))
    
    return (data=data, sites=sites, mbrs=mbrs, vars=vars, time_valid=tm, lts=lt, time_refs=tmr) 
end


#  read power production data
function read_production_data(; fname = "./data/wind_power_per_bidzone.csv")
    prod = CSV.read("./data/wind_power_per_bidzone.csv", DataFrame)
    dropmissing!(prod)
    rename!(prod, [:time, :NO1, :NO2, :NO3, :NO4])
    prod.time = DateTime.(prod.time, "yyyy-mm-dd HH:MM:SS")
    @views prod[!, 2:end] .= Float32.(prod[:, 2:end])

    return prod
end

#  read capacity information
function read_capacity_info(; fname = "./data/windparks_bidzone.csv")
    capacity = CSV.read(fname, DataFrame)[:, 1:4]  # omit eic code
    unique!(capacity)
    capacity.bidding_area = (u -> u[end-2:end]).(capacity.bidding_area)  # "ELSPOT N0*" -> "NO*"
    capacity.prod_start_new = DateTime.(capacity.prod_start_new, "yyyy-mm-dd HH:MM:SS")
    capacity.operating_power_max = Float32.(capacity.operating_power_max)
    
    return capacity
end



#  get capacity vector for given wind parks for each point in time
function get_capacity_vectors(capacity_df::DataFrame, parks::Vector{String}, tm::Vector{DateTime})
    out = zeros(Float32, length(parks), length(tm))
    park_dict = Dict(row.substation_name => 
        (row.operating_power_max, row.prod_start_new) for row in eachrow(capacity_df))
    for t in eachindex(tm)
        for p in eachindex(parks)
            prod, tm_start = park_dict[parks[p]]
            if tm[t] >= tm_start
                out[p, t] = prod
            end
        end
    end
    return out
end


function main()

    #  read MEPS forecast into 4d array of dimensions: site, member, variable, lt×time_ref
    @time meps = read_meps();  # ~3 minutes
    
    #  read wind power data
    prod = read_production_data()

    #  align data in time
    time_common = intersect(meps.time_valid, prod.time)
    idx_meps = occursin(meps.time_valid, time_common)
    x = meps.data[:, :, :, idx_meps]
    x_tm = meps.time_valid[idx_meps]

    idx_prod = occursin(prod.time, time_common)
    prod = prod[idx.prod, :]

    idx_prod = indexin(prod.time, x_tm)
    prod = prod[idx_prod, :]
    
    #  read wind park capacity information 
    capacity = read_capacity_info()
    @time capacity_vector = get_capacity_vectors(capacity, string.(meps.sites), meps.time_valid)

    
    #  create tuple for each price area, (x = c(fc, cap), y = prod)
    kt = indexin(meps.axes.time_valid, prod.time)
    ()
    
    #  save to JLD2
    
    
end
