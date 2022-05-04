module tt

using LKD
using JLD2
using Parameters
using Revise
using SpikeTimit
using ContentHashes


include("parameters.jl")
include("frame.jl")
include("voltage.jl")
include("triplet.jl")
include("input_layer.jl")
include("snn_layer.jl")
include("classification_layer.jl")
include("ml.jl")

greet() = print("Hello World!")

end # module
