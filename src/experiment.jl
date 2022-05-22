using LKD

@with_kw mutable struct Experiment
    name::String
end

function Experiment(name::String, params::LKD.InputParams)
    
end