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
    stdp::Union{tt.TripletSTDP, tt.VoltageSTDP}
end

function _getsid(id::String, stdp::Union{tt.TripletSTDP, tt.VoltageSTDP})
    stdp isa tt.VoltageSTDP ? "VOL_$(id)" : "TRI_$(id)"
end

function _rebuildfullsimpath(id, LR::Union{tt.TripletSTDP, tt.VoltageSTDP})
    if LR isa tt.TripletSTDP
        joinpath(tt.simsdir(), "TRI_"*id)
    else
        joinpath(tt.simsdir(), "VOL_"*id)
    end
end

function save(il::InputLayer)
    path = tt.rawdatadir()
    path = joinpath(path, "input_layers.jld2")
    new_data = Dict(
        "id" => il.id,
        "input_layer" => il
    );
    if !isfile(path)
        df = DataFrame(new_data)
    else
        df = JLD2.load(path, "input_layers");
        push!(df, new_data)
    end
    jldopen(path, "w") do file
        file["input_layers"] = df
    end;
end

function input_layer_exists(id::String)
    path = tt.rawdatadir()
    path = joinpath(path, "input_layers.jld2")
    if isfile(path)
        df = JLD2.load(path, "input_layers");
        if id in df[!,1]
            return true
        end
    end
    return false
end

function get_folder_name(params::LKD.InputParams, weights_params::LKD.WeightParams)
    "$(string(ContentHashes.hash(params)))_$(weights_params.Ne)_$(weights_params.Ni)";
end

function makeinputlayer(df::DataFrame, params::LKD.InputParams, weights_params::LKD.WeightParams, stdp = Union{tt.TripletSTDP, tt.VoltageSTDP})
    id = get_folder_name(params, weights_params);
    @info "Creating new input layer ($(id))."

    if stdp isa tt.VoltageSTDP
        folder_name = "VOL_$(id)"
    else
        folder_name = "TRI_$(id)"
    end    
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

function _loadinputlayer(id::String, stdp = Union{tt.TripletSTDP, tt.VoltageSTDP})
    @assert input_layer_exists(id) "Input layer does not exist."
    path = joinpath(tt.rawdatadir(), "input_layers.jld2")
    df = JLD2.load(path, "input_layers");
    fdf = filter(:id => x->x == id, df)
    il = fdf[!,2][1]
    il.id = id
    il.store.folder = _rebuildfullsimpath(id, stdp)
    il.stdp = stdp   
    il 
end


function InputLayer(df::DataFrame, params::LKD.InputParams, weights_params::LKD.WeightParams, stdp = Union{tt.TripletSTDP, tt.VoltageSTDP})
    id = get_folder_name(params, weights_params);
    if input_layer_exists(id)
        @info "Input layer exists, loading it."
        _loadinputlayer(id, stdp)
    else
        makeinputlayer(df, params, weights_params, stdp)
    end
end

"""
pass an isntance of TripletSTDP to triplet_stdp to use it, otherwise default is voltage-stdp
"""
function InputLayer(params::LKD.InputParams, weights_params::LKD.WeightParams, stdp = Union{tt.TripletSTDP, tt.VoltageSTDP})
    id = get_folder_name(params, weights_params);
    if stdp isa tt.VoltageSTDP
        folder_name = "VOL_$(id)"
    else
        folder_name = "TRI_$(id)"
    end    
    if input_layer_exists(id)
        @info "Input layer exists, loading it."
        path = tt.rawdatadir()
        path = joinpath(path, "input_layers.jld2")
        df = JLD2.load(path, "input_layers");
        fdf = filter(:id => x->x == id, df)
        il = fdf[!,2][1]
        il.id = id
        il.store.folder = _rebuildfullsimpath(id, stdp)
        il.stdp = stdp   
        il
    else
        df = tt.load_dataset(tt.datasetdir(), params);
        makeinputlayer(df, params, weights_params, stdp)
    end

end
