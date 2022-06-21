function is_neuron_in_assembly(neuron_id, popm)
    for i in 1:size(popm, 2)
        res = findall(x -> x == neuron_id, popm[(popm[:,i] .> -1), i])
        if length(res) > 0
            return true
        end
    end
    return false
end

function savesimids(expname, ids::Vector{String})
    p = joinpath(tt.rawdatadir(), "sim_ids.jld2")
    if isfile(p)
        rd = load(p)
        rd[expname] = ids
        save(p, rd)
    else
        save(p, Dict(expname => ids))
    end
end

function savesimid(expname, id::String)
    p = joinpath(tt.rawdatadir(), "sim_ids.jld2")
    if isfile(p)
        rd = load(p)
        rd[expname] = [id]
        save(p, rd)
    else
        save(p, Dict(expname => [id]))
    end
end