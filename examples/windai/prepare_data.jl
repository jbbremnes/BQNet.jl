#   read and prepare data
#
#

using DataFrames, Dates
using CSV, JLD2


#  read MEPS data from CSV file and convert to 5d array
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
    append!(vars, "leadtime")

    #  reduce to 4d array here!
    #meps = reshape(meps, size(meps,1), size(meps,2), size(meps,3), :)
    tm = vec(Hour.(lts) .+ permutedims(time_refs))
    lt = repeat(lts, outer = length(time_refs))
    tmr = repeat(time_refs, inner = length(lts))
    
    return (data=data, axes=(sites=sites, mbrs=mbrs, vars=vars, lts=lts, time_refs=time_refs)) # fix here!
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



function main()

    #  read MEPS forecast into 5d array of dimensions: site, member, variable, lt, time_ref
    @time meps = read_meps()  # ~3 minutes
    
    #  read wind power data
    prod = read_production_data()
    
    #  read wind park capacity information 
    capacity = read_capacity_info()
    meps[2].time_refs <
 
    #  create tuple for each price area, (x = c(fc, cap), y = prod)
    
    
    
    #  save to JLD2
    
    
end
