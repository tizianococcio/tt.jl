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

### io
function preparefolder(folder::String, 
    keep::Vector{String}=["trials", "weights", "network.h5", "output.jld2"])
    for p in readdir(folder)
        if p in keep
            continue
        end
        rm(joinpath(folder, p), recursive=true)
    end
    mkdir(joinpath(folder, "mean_weights"))
    mkdir(joinpath(folder, "word_states"))
    mkdir(joinpath(folder, "phone_states"))
    mkdir(joinpath(folder, "spikes"))    
end

function savexpdata(exp_name, what; kargs...)
    JLD2.jldsave(joinpath(tt.processeddatadir(), "$(exp_name)_$(what).jld2"); kargs...)
end

function loadexpdata(exp_name, what)
    jldopen(joinpath(tt.processeddatadir(), "$(exp_name)_$(what).jld2"))
end

function loadexpobject(filename::String)
    load_object(joinpath(tt.processeddatadir(), filename))
end
function loadexpobject(exp_name, what)
    load_object(joinpath(tt.processeddatadir(), "$(exp_name)_$(what).jld2"))
end

function get_weight_files_list(in::InputLayer)
    folder = joinpath(in.store.folder, "weights")
    files = []
    for file_ in readdir(folder)
        if startswith(file_,"weights")
            filename = joinpath(folder,file_)
            tt.LKD.h5open(filename,"r") do fid
                tt = read(fid["t"])
                push!(files,(tt, filename))
            end
        end
        sort!(files,by=x->x[1])
    end
    files
end

function save_compressed_weights(W, t, folder)
    jldsave(joinpath(folder, "weights", "weights_T$(round(Int,t))"), true; weights=W, t=t)
end