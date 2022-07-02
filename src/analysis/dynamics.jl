using StatsBase

struct WeightTrace
    count::Int
    paths::Vector{String}
end
function _readweights(file)
    tt.LKD.h5open(file,"r") do file_
        fid = read(file_)
        W = Array(fid["weights"])    
        return W
    end
end
Base.iterate(S::WeightTrace, state=1) = state > S.count ? nothing : (_readweights(S.paths[state]), state+1)
Base.firstindex(S::tt.WeightTrace) = 1
Base.lastindex(S::tt.WeightTrace) = length(S)

function Base.getindex(S::WeightTrace, i::Int)
    1 <= i <= S.count || throw(BoundsError(S, i))
    return _readweights(S.paths[i])
end
Base.length(S::WeightTrace) = S.count

"""
Computes KL-divergence of two weight traces
1. Between the first init weights and the weights at time test (first_vs_t)
2. Between the weight at time t and the final weight (t_vs_end)

trace: the weight trace as returned by tt.get_weight_traces()
n: the number of neurons to consider

Returns a tuple with two vectors of divergences
"""
function compute_klds(trace, n)
    t = length(trace)
    fine = t - 1
    first_vs_t = [
        kldivergence(trace[1][1:n,1:n][:], trace[i][1:n,1:n][:])
            for i in 1:t
    ]

    t_vs_end = [
        kldivergence(trace[i][1:n,1:n][:], trace[fine][1:n,1:n][:])
            for i in 1:t
    ]
    first_vs_t, t_vs_end
end

"""
version that takes an iterator. mostly used to load many files without using a lot of memory
"""
function compute_klds(trace::WeightTrace, n)
    t = trace.count
    fine = t - 1
    _start = trace[1][1:n,1:n][:]
    _end = trace[trace.count][1:n,1:n][:]
    first_vs_t = Vector{Float64}(undef, t)
    t_vs_end = Vector{Float64}(undef, t)
    for (i, w) in enumerate(trace)
        first_vs_t[i] = kldivergence(_start, w[1:n,1:n][:])
        t_vs_end[i] = kldivergence(w[1:n,1:n][:], _end)
    end
    first_vs_t, t_vs_end
end