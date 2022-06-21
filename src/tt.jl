module tt

using LKD
using JLD2
using Parameters
using Revise
using SpikeTimit
using ContentHashes
using StatsBase
using Dates

include("parameters.jl")
include("frame.jl")
include("voltage.jl")
include("voltage_m.jl")
include("triplet.jl")
include("triplet_traces.jl")
include("triplet_det.jl")
include("triplet_traces_alt.jl") # decreases detectors only if there is no spikes
include("triplet_traces_full.jl")
include("triplet_nostdp.jl")
include("triplet_barebones.jl")
include("input_layer.jl")
include("snn_layer.jl")
include("classification_layer.jl")
include("experiment.jl")
include("evaluation.jl")
include("utils.jl")

greet() = print("Hello World!")

end # module
