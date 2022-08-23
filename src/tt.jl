module tt

using LKD
using JLD2, CodecZlib
using Parameters
using Revise
using SpikeTimit
using ContentHashes
using StatsBase
using Dates
using Plots

abstract type TrackersT end
struct Trackers <: TrackersT
    Voltage::Matrix{Float64}
    AdaptCurrent::Matrix{Float64}
    AdaptThresh::Matrix{Float64}
    WeightPre::Vector{Matrix{Float64}} # each matrix = Nsteps x Nsynapses
    WeightPost::Vector{Matrix{Float64}}
    r₁::Matrix{Float64}
    r₂::Matrix{Float64}
    o₁::Matrix{Float64}
    o₂::Matrix{Float64}
end


Plots.default(
    size = (1000,600),
    framestyle = :box,    
    titlefont=font(20,"Computer Modern"),
    xtickfont=font(14,"Computer Modern"),
    ytickfont=font(14,"Computer Modern"),
    guidefont=font(14,"Computer Modern")
)

include("parameters.jl")
include("frame.jl")
include("voltage.jl")
include("voltage_m.jl")
include("triplet.jl")
include("triplet_traces.jl")
include("triplet_traces_alt.jl")
include("triplet_nostdp.jl")
include("input_layer.jl")
include("snn_layer.jl")
include("classification_layer.jl")
include("experiment.jl")
include("evaluation.jl")
include("utils.jl")
include("analysis/dynamics.jl")
include("plots.jl")

end # module
