using DataFrames
using Random
using SpikeTimit
using LKD
using YAML
using MacroTools
using Makie

function load_conf()
    return YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
end

function rawdatadir()
    c = load_conf()
    joinpath(c["data"], "exp_raw")
end

function processeddatadir()
    c = load_conf()
    joinpath(c["data"], "exp_pro")
end

function datasetdir()
    c = load_conf()
    c["dataset_path"]
end

function simsdir()
    c = load_conf()
    c["training_storage_path"]
end

function plotsdir()
    c = load_conf()
    c["plots"]
end

function saveplot(filename::String, f::Union{Figure, Plots.Plot{Plots.GRBackend}}; kargs...)
    saveplot(f, filename; kargs...)
end

function saveplot(f::Union{Figure, Plots.Plot{Plots.GRBackend}}, filename::String; kargs...)
    if f isa Plots.Plot{Plots.GRBackend}
        savefig(f, joinpath(tt.plotsdir(), filename); kargs...)
    else
        Makie.save(joinpath(tt.plotsdir(), filename), f; kargs...)
    end
end

function get_timit_train_dataframe()
    _get_timit_dataframe(datasetdir())
end
function get_timit_test_dataframe()
    _get_timit_dataframe(datasetdir(), which="test")
end

function get_timit_train_dataframe(path::String)
    _get_timit_dataframe(path)
end
function get_timit_test_dataframe(path::String)
    _get_timit_dataframe(path, which="test")
end

function get_dialectdict()
    Dict(
        1 => "New England",
        2 => "Northern",
        3 => "North Midland",
        4 => "South Midland",
        5 => "Southern",
        6 => "New York City",
        7 => "Western",
        8 => "Moved around"
    )
end

function _get_timit_dataframe(path::String; which="train")
    cache_path = joinpath(path, "cache");
    cached_df = joinpath(cache_path, "$which.jld2");
    input_df = DataFrame();
    if !isdir(cache_path)
        mkpath(cache_path);
    end
    if !isfile(cached_df)
        @info "Creating cached dataframe";
        jldopen(cached_df, "w") do file
            input_df = SpikeTimit.create_dataset(;dir=joinpath(path, which));
            file[which] = input_df;
        end;
    else
        @info "Reading cached dataframe";
        input_df = JLD2.load(cached_df, which);
    end
    return input_df
end

function getdictionary()
    dict_path = joinpath(datasetdir(), "DOC", "TIMITDIC.TXT")
    SpikeTimit.create_dictionary(file=dict_path)
end

function load_dataset(path, params::LKD.InputParams)
    input_df = get_timit_train_dataframe(path);
    filtered_df = filter(
        :words => x-> any([word in x for word in params.words]), input_df) |> 
        df -> filter(:dialect => x->x ∈ params.dialects, df) |> 
        df -> filter(:gender => x->x ∈ params.gender, df)
    return filtered_df
end

function load_network(dataframe::DataFrame, params::LKD.InputParams, training_storage_path)

    word_inputs = SpikeTimit.select_words(
        dataframe, 
        samples=params.samples, 
        params.words, 
        encoding=params.encoding)

    if params.encoding == "bae"
        SpikeTimit.resample_spikes!(word_inputs.spikes)
        SpikeTimit.transform_into_bursts!(word_inputs.spikes)
    end
    
    Random.seed!(params.random_seed)

    # Mix the inputs (3 repetitions->why 3?)
    shuffled_ids = SpikeTimit.mix_inputs(length(word_inputs.durations), params.repetitions)
    spikes, transcriptions = SpikeTimit.get_ordered_spikes(
        word_inputs, 
        ids=shuffled_ids, 
        silence_time=params.silence_time, 
        shift = params.shift_input)
    
    ## Get transcriptions and firing time for timesteps
    transcriptions_dt = SpikeTimit.transcriptions_dt(transcriptions)
    spikes_dt = SpikeTimit.ft_dt(spikes)
    
    # Set simulation details
    last_interval = ceil(transcriptions.phones.intervals[end][end]*1000)
    net_params = LKD.NetParams(simulation_time = last_interval, learning=true)
    
    # Set the input weights and projections parameters
    weights_params = LKD.WeightParams(
        Ne = Ne,
        Ni = Ni
    )

    projections = LKD.ProjectionParams(
        npop = SpikeTimit.get_encoding_dimension(word_inputs.spikes)
    )
            
    # Create and store network
    store = LKD.StoreParams(
        folder = training_storage_path, 
        save_timestep=10_000)

    return Dict(
        "weights_matrix" => W,
        "population_matrix" => popmembers,
        "firing_times" => spikes_dt,
        "transcriptions" => transcriptions_dt, # timestamps (start,stop) of each word
        "network_params" => net_params,
        "storage_params" => store,
        "weights_params" => weights_params,
        "projection_params" => projections
    )
    
end

function prepare_network(dataframe::DataFrame, params::LKD.InputParams, training_storage_path; Ne=4000, Ni=1000)
    word_inputs = SpikeTimit.select_words(
        dataframe, 
        samples=params.samples, 
        params.words, 
        encoding=params.encoding)

    if params.encoding == "bae"
        SpikeTimit.resample_spikes!(word_inputs.spikes)
        SpikeTimit.transform_into_bursts!(word_inputs.spikes)
    end
    
    Random.seed!(params.random_seed)

    # Mix the inputs
    shuffled_ids = SpikeTimit.mix_inputs(length(word_inputs.durations), params.repetitions)
    spikes, transcriptions = SpikeTimit.get_ordered_spikes(
        word_inputs, 
        ids=shuffled_ids, 
        silence_time=params.silence_time, 
        shift = params.shift_input)
    
    ## Get transcriptions and firing time for timesteps
    transcriptions_dt = SpikeTimit.transcriptions_dt(transcriptions)
    spikes_dt = SpikeTimit.ft_dt(spikes)
    
    # Set simulation details
    last_interval = ceil(transcriptions.phones.intervals[end][end]*1000)
    net_params = LKD.NetParams(simulation_time = last_interval, learning=true)
    
    # Set the input weights and projections parameters
    weights_params = LKD.WeightParams(
        Ne = Ne,
        Ni = Ni
    )

    projections = LKD.ProjectionParams(
        npop = SpikeTimit.get_encoding_dimension(word_inputs.spikes)
    )
            
    # Create and store network
    store = LKD.StoreParams(
        folder = training_storage_path, 
        save_timestep=10_000)
    
    W, popmembers = LKD.create_network(weights_params, projections)
    
    return Dict(
        "weights_matrix" => W,
        "population_matrix" => popmembers,
        "firing_times" => spikes_dt,
        "transcriptions" => transcriptions_dt, # timestamps (start,stop) of each word
        "network_params" => net_params,
        "storage_params" => store,
        "weights_params" => weights_params,
        "projection_params" => projections
    )

end

"""
save_network(weight_matrix, population_matrix, storage_params, hard::Bool=False)

hard::Bool if True wipes the destination folder before saving
"""
function save_network(weight_matrix, population_matrix, storage_params, hard::Bool=False)
    folder = LKD.makefolder(storage_params.folder);
    if hard
        folder = LKD.cleanfolder(storage_params.folder);
    end
    LKD.save_network(population_matrix, weight_matrix, storage_params.folder)
end

function save_weight_traces(folder, pre, post)
    file = jldopen(joinpath(folder, "w_traces.jld2"), "w")
    file["pre"] = pre
    file["post"] = post
    close(file)
end

function load_weight_traces(folder)
    file = jldopen(joinpath(folder, "w_traces.jld2"), "r")
    pre = file["pre"]
    post = file["post"]
    close(file)
    pre, post
end

function save_weight_traces(folder, container::Vector{Matrix{Float64}})
    file = jldopen(joinpath(folder, "w_traces_full.jld2"), "w")
    file["container"] = container
    close(file)
end

function load_fullmat_weight_traces(folder)
    file = jldopen(joinpath(folder, "w_traces_full.jld2"), "r")
    c = file["container"]
    close(file)
    c
end

"""
from DrWatson
https://github.com/JuliaDynamics/DrWatson.jl/blob/599a9b2c04837e9d2162a022baf3394376af0cd9/src/naming.jl
"""
macro strdict(vars...)
    expr = Expr(:call, :Dict)
	for var in vars
		# Allow assignment syntax a = b
		if @capture(var, a_ = b_)
			push!(expr.args, :($(string(a)) => $(esc(b))))
		# Allow single arg syntax a   → "a" = a
		elseif @capture(var, a_Symbol)
			push!(expr.args, :($(string(a)) => $(esc(a))))
		else
			return :(throw(ArgumentError("Invalid field syntax")))
		end
	end
	return expr
end

function extrdict(dict)
    t_ins = dict["t_ins"]
    v_ins = dict["v_ins"]
    t_outs = dict["t_outs"]
    v_outs = dict["v_outs"]
    t_ins, v_ins, t_outs, v_outs
end

function savedata(t_ins, v_ins, t_outs, v_outs, name)
    data = @tt.strdict(t_ins, v_ins, t_outs, v_outs)
    e = tt.Experiment(name, "")
    tt.save(data, e)
end

function loaddata(name)
    e = tt.Experiment(name, "")
    extrdict(e.df[1,:].data)
end