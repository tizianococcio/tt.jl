@with_kw mutable struct InputLayer
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

function get_folder_name(params::LKD.InputParams, weights_params::LKD.WeightParams)
    "$(string(ContentHashes.hash(params)))_$(weights_params.Ne)_$(weights_params.Ni)";
end

"""
pass an isntance of TripletSTDP to triplet_stdp to use it, otherwise default is voltage-stdp
"""
function InputLayer(params::LKD.InputParams, weights_params::LKD.WeightParams, path_dataset, path_storage, stdp = Union{tt.TripletSTDP, tt.VoltageSTDP})
    df = tt.load_dataset(path_dataset, params);
    word_inputs = SpikeTimit.select_words(
        df, 
        samples=params.samples, 
        params.words, 
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
    # Naming convention: LearningRule_Encoding_NumWords_d(NumDialects)s(NumSamples)r(NumRepetitions)sh(ShiftInput)g(NumGenders)_Ne_Ni
    # filename_inputs = "$(params.encoding)_$(length(params.words))_d$(length(params.dialects))s$(params.samples)r$(params.repetitions)sh$(round(Int, params.shift_input))g$(length(params.gender))_$(weights_params.Ne)_$(weights_params.Ni)"
    filename_inputs = "$(string(ContentHashes.hash(params)))_$(weights_params.Ne)_$(weights_params.Ni)";
    # to save the post-training network
    if stdp isa tt.VoltageSTDP
        folder_name = "VOL_$(filename_inputs)"
    else
        folder_name = "TRI_$(filename_inputs)"
    end

    store = LKD.StoreParams(folder = joinpath(path_storage, folder_name), save_states=true, save_network=true, save_weights=true);

    InputLayer(
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
end
