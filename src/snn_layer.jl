@with_kw mutable struct SNNLayer
    weights::Matrix{Float64}
    popmembers::Matrix{Int64}
    spikes_dt#::SpikeTimit.FiringTimes,
    transcriptions_dt::SpikeTimit.Transcriptions
    net::LKD.NetParams
    store::LKD.StoreParams
    weights_params::LKD.WeightParams
    projections::LKD.ProjectionParams
    stdp::Union{tt.TripletSTDP, tt.VoltageSTDP}
end

function SNNLayer(in::tt.InputLayer)
    SNNLayer(
        in.weights,
        in.popmembers,
        in.spikes_dt,
        in.transcriptions_dt,
        in.net,
        in.store,
        in.weights_params,
        in.projections,
        in.stdp
    )
end

# (voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker, r1, o1, r2, o2, weight_tracker)
trackers_triplet = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Matrix{Float64}}
# (voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker, u, v, weight_tracker)
trackers_voltage = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Matrix{Float64}}
@with_kw struct SNNOut
    weights::Matrix{Float64}
    firing_times::Vector{Vector{Float64}}
    firing_rates::Matrix{Float32}
    trackers:: Union{trackers_voltage, trackers_triplet} 
    phone_states::Vector{Any}
    word_states::Vector{Any}
end

function _run(snn::SNNLayer, traces::Bool=false)
    if traces
        W, T, R, trackers = tt.sim_m(snn.weights, snn.popmembers, snn.spikes_dt, snn.transcriptions_dt, 
            snn.net, snn.store, snn.weights_params, snn.projections, snn.stdp);
    else
        W, T, R, trackers = tt.sim(snn.weights, snn.popmembers, snn.spikes_dt, snn.transcriptions_dt, 
            snn.net, snn.store, snn.weights_params, snn.projections, snn.stdp);
    end
    ws = LKD.read_network_states(joinpath(snn.store.folder,"word_states"));
    ps = LKD.read_network_states(joinpath(snn.store.folder,"phone_states"));
    return SNNOut(W, T, R, trackers, ps, ws)
end

"""
runs the simulation and stores the network states. Overwrite previous data.
"""
function train(snn::SNNLayer)
    LKD.makefolder(snn.store.folder);
    LKD.cleanfolder(snn.store.folder);
    return _run(snn)
end

function train_with_traces(snn::SNNLayer)
    LKD.makefolder(snn.store.folder);
    LKD.cleanfolder(snn.store.folder);
    return _run(snn, true)
end


"""
runs the simulation on a previously trained network. Data for this network must exist on disk.
"""
function test(snn::SNNLayer)
    snn.net.learning = false
    snn.store.save_states=false
    snn.store.save_network=false
    snn.store.save_weights=false
    return _run(snn)
end

"""
loads network from disk and updates `in` with the stored weights
"""
function load(in::tt.InputLayer)
    W = LKD.read_network_weights(in.store.folder)
    T = LKD.read_network_spikes(in.store.folder)
    R = LKD.read_network_rates(in.store.folder)
    SS_words = LKD.read_network_states(joinpath(in.store.folder,"word_states"))
    SS_phones = LKD.read_network_states(joinpath(in.store.folder,"phone_states"))
    voltage_tracker = LKD.read_neuron_membrane(in.store.folder)
    adaptation_current_tracker = LKD.read_neuron_membrane(in.store.folder; type="w_adapt")
    adaptive_threshold_tracker = LKD.read_neuron_membrane(in.store.folder; type="adaptive_threshold")
    trackers = voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker
    in.weights = W[end][2]
    return (snn_layer=SNNLayer(in), states=(weights=W, firing_times=T, firing_rates=R, phone_states=SS_phones, word_states=SS_words, trackers=trackers))
end

"""
inject input from new input layer into a pretrained network layer
"""
function inject(new::tt.InputLayer, old::tt.SNNLayer)
    old.spikes_dt = new.spikes_dt
    old.transcriptions_dt = new.transcriptions_dt
    new.net.simulation_time = old.net.simulation_time # copy simulation time over to be used in the classifier
    return old
end