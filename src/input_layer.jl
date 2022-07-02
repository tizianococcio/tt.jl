@with_kw mutable struct InputLayer
    id::String
    weights::Matrix{Float64}
    popmembers::Matrix{Int64}
    spikes_dt#::SpikeTimit.FiringTimes,
    transcriptions::SpikeTimit.Transcriptions
    transcriptions_dt::SpikeTimit.Transcriptions
    net::LKD.NetParams
    store::LKD.StoreParams
    weights_params::LKD.WeightParams
    projections::LKD.ProjectionParams
    stdp::STDP
end

function _rebuildfullsimpath(id)
    joinpath(tt.simsdir(), id)
end

function _rebuildfullsimpath(subfolder::String, id)
    joinpath(tt.simsdir(), subfolder, id)
end

function save(il::InputLayer, params::LKD.InputParams)
    if !input_layer_exists(il.id)
        ils_folder = joinpath(tt.rawdatadir(), "input_layers");
        if !isdir(ils_folder)
            mkdir(ils_folder)
        end
        jldsave(joinpath(ils_folder, "$(il.id).jld2"), layer=il, params=params);
    else
        @info "Input layer $(il.id) already exists, cannot save it. Returning it."
        il
    end
end

function delete(il::InputLayer)
    if input_layer_exists(il.id)
        ils_folder = joinpath(tt.rawdatadir(), "input_layers");
        rm(joinpath(ils_folder, "$(il.id).jld2"))
    end
end

function _loadinputlayer(id::String)
    @assert input_layer_exists(id) "Input layer does not exist."
    d = JLD2.load(joinpath(tt.rawdatadir(), "input_layers", "$(id).jld2"))
    il = d["layer"]
    il.id = id
    il.store.folder = _rebuildfullsimpath(id)
    il, d["params"]
end

function _loadinputlayer(id::String, stdp::STDP)
    @assert input_layer_exists(id) "Input layer does not exist."
    ils_folder = joinpath(tt.rawdatadir(), "input_layers");
    d = JLD2.load(joinpath(ils_folder, "$(id).jld2"))
    il = d["layer"]
    il.id = id
    il.store.folder = _rebuildfullsimpath(id)
    il.stdp = stdp   
    il, d["params"]
end

function input_layer_exists(id::String)
    isfile(joinpath(tt.rawdatadir(), "input_layers", "$(id).jld2"))
end

function get_folder_name(params::LKD.InputParams, weights_params::LKD.WeightParams)
    "$(string(ContentHashes.hash(params)))_$(weights_params.Ne)_$(weights_params.Ni)";
end

function get_folder_name(params::LKD.InputParams, weights_params::LKD.WeightParams, stdp::STDP)
    "$(string(ContentHashes.hash([params, weights_params, stdp])))_$(weights_params.Ne)_$(weights_params.Ni)";
end

function makeinputlayer(df::DataFrame, params::LKD.InputParams, weights_params::LKD.WeightParams, stdp::STDP; save = true, subfolder = "")
    id = get_folder_name(params, weights_params, stdp);
    @info "Creating new input layer ($(id))."
    folder_name = id
    path_storage = tt.simsdir()
    if length(subfolder) > 0
        path_storage = joinpath(tt.simsdir(), subfolder)
        @assert isdir(path_storage) "Storage path does not exist"
    end
    word_inputs = SpikeTimit.select_words(
        df, 
        params.words; 
        samples=params.samples, 
        encoding=params.encoding);
    if params.encoding == "bae"
        SpikeTimit.resample_spikes!(word_inputs.spikes)
        SpikeTimit.transform_into_bursts!(word_inputs.spikes)
    end
    ids = tt.SpikeTimit.mix_inputs(length(word_inputs.durations), params.repetitions);
    ordered_spikes, transcripts = SpikeTimit.get_ordered_spikes(word_inputs, ids=ids, silence_time=params.silence_time, shift=params.shift_input);
    transcriptions_dt = SpikeTimit.transcriptions_dt(transcripts);
    spikes_dt = SpikeTimit.ft_dt(ordered_spikes);
    projections = LKD.ProjectionParams(npop = SpikeTimit.get_encoding_dimension(word_inputs.spikes));
    W, popmembers = LKD.create_network(weights_params, projections);
    last_interval = ceil(transcripts.phones.intervals[end][end]*1000);
    net = LKD.NetParams(simulation_time = last_interval, learning=true);
    store = LKD.StoreParams(folder = joinpath(path_storage, folder_name), save_states=true, save_network=true, save_weights=true);

    il = InputLayer(
        id,
        W,
        popmembers,
        spikes_dt,
        transcripts,
        transcriptions_dt,
        net,
        store,
        weights_params,
        projections,
        stdp
    )
    if save
        tt.save(il, params)
    end
    il
end

"""
shuffles order of the words to be given as input to the network. 
"""
function shuffle_words(il::InputLayer)
    _, params = _loadinputlayer(il.id)
    df = tt.load_dataset(tt.datasetdir(), params);

    word_inputs = SpikeTimit.select_words(
        df, 
        params.words; 
        samples=params.samples, 
        encoding=params.encoding);
    if params.encoding == "bae"
        SpikeTimit.resample_spikes!(word_inputs.spikes)
        SpikeTimit.transform_into_bursts!(word_inputs.spikes)
    end    
    ids = tt.SpikeTimit.mix_inputs(length(word_inputs.durations), params.repetitions);
    ordered_spikes, transcripts = SpikeTimit.get_ordered_spikes(word_inputs, ids=ids, silence_time=params.silence_time, shift=params.shift_input);
    transcriptions_dt = SpikeTimit.transcriptions_dt(transcripts);
    spikes_dt = SpikeTimit.ft_dt(ordered_spikes);
    il_shuffle = deepcopy(il)
    il_shuffle.spikes_dt = spikes_dt
    il_shuffle.transcriptions = transcripts
    il_shuffle.transcriptions_dt = transcriptions_dt
    il_shuffle, [w.word for w in word_inputs.labels[ids]]
end




function InputLayer(df::DataFrame, params::LKD.InputParams, weights_params::LKD.WeightParams, stdp::STDP)
    id = get_folder_name(params, weights_params, stdp);
    if input_layer_exists(id)
        @info "Input layer $(id) exists, loading it."
        il, _ = _loadinputlayer(id, stdp)
        il
    else
        makeinputlayer(df, params, weights_params, stdp)
    end
end

"""
pass an isntance of TripletSTDP to triplet_stdp to use it, otherwise default is voltage-stdp
"""
function InputLayer(params::LKD.InputParams, weights_params::LKD.WeightParams, stdp::STDP; save = true, subfolder="")
    id = get_folder_name(params, weights_params, stdp);  
    if input_layer_exists(id)
        @info "Input layer ($(id)) exists, loading it."
        il, _ = _loadinputlayer(id, stdp)
        il
    else
        df = tt.load_dataset(tt.datasetdir(), params);
        makeinputlayer(df, params, weights_params, stdp; save=save, subfolder=subfolder)
    end

end


function makeinput(in_params::tt.LKD.InputParams, weight_params::tt.LKD.WeightParams, stdp::STDP=tt.TripletSTDP(); save=true, subfolder="")

    # start from triplet
    tri = tt.InputLayer(in_params, weight_params, stdp; save=save, subfolder=subfolder);

    # copy into voltage
    vol = deepcopy(tri)
    vol.stdp = tt.VoltageSTDP()
    vol.id =  tt.get_folder_name(in_params, weight_params, tt.VoltageSTDP())
    vol.store.folder = tt._rebuildfullsimpath(subfolder, vol.id)
    if save
        tt.save(vol, in_params)
    end
    
    tri, vol
end

function newlike(il::InputLayer, params::tt.LKD.InputParams; wp=nothing, new_stdp=nothing, save=true, subfolder="")
        new = deepcopy(il)
        if !isnothing(new_stdp)
            new.stdp = new_stdp
        end
        if !isnothing(wp)
            new.weight_params = wp
        end
        new.id =  tt.get_folder_name(params, new.weights_params, new.stdp)
        new.store.folder = tt._rebuildfullsimpath(subfolder, new.id)
        @info "New layer id $(new.id)"
        if save
            tt.save(new, params)
        end
        new
end

"""
makes simulation longer of n seconds
returns tuple with (old sim time, new sim time)
"""
function extend_simtime(il::InputLayer, n::Int)
    old = il.net.simulation_time
    extra_time = n*1000 #ms
    il.net.simulation_time += extra_time
    old, il.net.simulation_time
end