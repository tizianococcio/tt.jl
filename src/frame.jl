# old versions of wrapper code

using DataFrames
using Random
using SpikeTimit
using LKD

function load_conf()
    return YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
end

function load_dataset(path, params::LKD.InputParams)
    train_path = joinpath(path, "train");
    train = SpikeTimit.create_dataset(;dir= train_path)

    filtered_df = filter(
        :words => x-> any([word in x for word in params.words]), train) |> 
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
