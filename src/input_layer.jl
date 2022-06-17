@with_kw mutable struct InputLayer
    id::String
    # sid::String
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

function _getsid(id::String, stdp::STDP)
    stdp isa tt.VoltageSTDP ? "VOL_$(id)" : "TRI_$(id)"
end

function _rebuildfullsimpath(id)
    joinpath(tt.simsdir(), id)
end

function save(il::InputLayer)
    if !input_layer_exists(il.id)
        ils_folder = joinpath(tt.rawdatadir(), "input_layers");
        if !isdir(ils_folder)
            mkdir(ils_folder)
        end
        save_object(joinpath(ils_folder, "$(il.id).jld2"), il);
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
    ils_folder = joinpath(tt.rawdatadir(), "input_layers");
    il = load_object(joinpath(ils_folder, "$(id).jld2"))
    il.id = id
    il.store.folder = _rebuildfullsimpath(id)
    il 
end

function _loadinputlayer(id::String, stdp::STDP)
    @assert input_layer_exists(id) "Input layer does not exist."
    ils_folder = joinpath(tt.rawdatadir(), "input_layers");
    il = load_object(joinpath(ils_folder, "$(id).jld2"))
    il.id = id
    il.store.folder = _rebuildfullsimpath(id)
    il.stdp = stdp   
    il 
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

function makeinputlayer(df::DataFrame, params::LKD.InputParams, weights_params::LKD.WeightParams, stdp::STDP)
    id = get_folder_name(params, weights_params, stdp);
    @info "Creating new input layer ($(id))."
    folder_name = id
    path_storage = tt.simsdir()
    word_inputs = SpikeTimit.select_words(
        df, 
        params.words; 
        samples=params.samples, 
        encoding=params.encoding);
    if params.encoding == "bae"
        SpikeTimit.resample_spikes!(word_inputs.spikes)
        SpikeTimit.transform_into_bursts!(word_inputs.spikes)
    end
    ids = SpikeTimit.mix_inputs(length(word_inputs.durations), params.repetitions);
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
        # _getsid(id, stdp),
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
    tt.save(il)
    il
end




function InputLayer(df::DataFrame, params::LKD.InputParams, weights_params::LKD.WeightParams, stdp::STDP)
    id = get_folder_name(params, weights_params, stdp);
    if input_layer_exists(id)
        @info "Input layer $(id) exists, loading it."
        _loadinputlayer(id, stdp)
    else
        makeinputlayer(df, params, weights_params, stdp)
    end
end

"""
pass an isntance of TripletSTDP to triplet_stdp to use it, otherwise default is voltage-stdp
"""
function InputLayer(params::LKD.InputParams, weights_params::LKD.WeightParams, stdp::STDP)
    id = get_folder_name(params, weights_params, stdp);  
    if input_layer_exists(id)
        @info "Input layer ($(id)) exists, loading it."
        _loadinputlayer(id, stdp)
    else
        df = tt.load_dataset(tt.datasetdir(), params);
        makeinputlayer(df, params, weights_params, stdp)
    end

end


function makeinput(in_params::tt.LKD.InputParams, weight_params::tt.LKD.WeightParams, stdp::STDP=tt.TripletSTDP())

    # start from triplet
    tri = tt.InputLayer(in_params, weight_params, stdp);

    # copy into voltage
    vol = deepcopy(tri)
    vol.stdp = tt.VoltageSTDP()
    vol.id =  tt.get_folder_name(in_params, weight_params, tt.VoltageSTDP())
    vol.store.folder = tt._rebuildfullsimpath(vol.id)
    tt.save(vol)
    
    tri, vol
end

function newlike(il::InputLayer, params::tt.LKD.InputParams; wp=nothing, new_stdp=nothing)
        new = deepcopy(il)
        if !isnothing(new_stdp)
            new.stdp = new_stdp
        end
        if !isnothing(wp)
            new.weight_params = wp
        end
        new.id =  tt.get_folder_name(params, new.weights_params, new.stdp)
        new.store.folder = tt._rebuildfullsimpath(new.id)
        @info "New layer id $(new.id)"
        tt.save(new)
        new
end