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

# (voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker)
trackers_triplet_basic = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
# (voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker, r1, o1, r2, o2, weight_tracker)
trackers_triplet = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Tuple{Matrix{Float64}, Matrix{Float64}}}
# (voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker, u, v, weight_tracker)
trackers_voltage = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Tuple{Matrix{Float64}, Matrix{Float64}}}
@with_kw struct SNNOut
    weights::Matrix{Float64}
    firing_times::Vector{Vector{Float64}}
    firing_rates::Matrix{Float32}
    trackers:: Union{trackers_voltage, trackers_triplet_basic, trackers_triplet} 
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
function train(snn::SNNLayer; overwrite=false)

    if isdir(snn.store.folder)
        if overwrite
            LKD.makefolder(snn.store.folder);
            LKD.cleanfolder(snn.store.folder);    
            return _run(snn)
        else
            # load trained data and return it
            @info "A trained network already exists. Now loading it. To wipe it and run a new training pass overwrite=true."
            return load(snn.store.folder)
        end
    else
        LKD.makefolder(snn.store.folder);
        LKD.cleanfolder(snn.store.folder);
        return _run(snn)
    end

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
    if !isdir(snn.store.folder)
        mkdir(snn.store.folder)
    end
    snn.store.folder = joinpath(snn.store.folder, string(now()))
    LKD.makefolder(snn.store.folder)
    snn.store.save_states=true
    snn.store.save_network=true
    snn.store.save_weights=false
    @info snn.store
    return _run(snn)
end

"""
loads network from disk and updates `in` with the stored weights
"""
function load(in::tt.InputLayer)
    datafile = joinpath(in.store.folder, "output.jld2")
    SS_words = LKD.read_network_states(joinpath(in.store.folder,"word_states"))
    SS_phones = LKD.read_network_states(joinpath(in.store.folder,"phone_states"))
    if isfile(datafile)
        f = jldopen(datafile, "r")
        W_last = f["weights"]
        T = f["spikes"]
        R = f["rates"]
        voltage_tracker = f["voltage_tracker"]
        adaptation_current_tracker = f["adaptation_current_tracker"]
        adaptive_threshold_tracker = f["adaptive_threshold_tracker"]
        if haskey(f, "r1") && haskey(f, "weight_tracker_pre")
            r1 = f["r1"]
            r2 = f["r2"]
            o1 = f["o1"]
            o2 = f["o2"]
            weight_tracker_pre = f["weight_tracker_pre"]
            weight_tracker_post = f["weight_tracker_post"]
            trackers = voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker, r1, o1, r2, o2, (weight_tracker_pre, weight_tracker_post)
        end
        
        if haskey(f, "u_trace") && haskey(f, "weight_tracker_pre")
            u_trace = f["u_trace"]
            v_trace = f["v_trace"]
            weight_tracker_pre = f["weight_tracker_pre"]
            weight_tracker_post = f["weight_tracker_post"]
            trackers = voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker, u_trace, v_trace, (weight_tracker_pre, weight_tracker_post)
        end
        close(f)
        w_trace = LKD.read_network_weights(in.store.folder)
    else
        w_trace = LKD.read_network_weights(in.store.folder)
        T = LKD.read_network_spikes(in.store.folder)
        R = LKD.read_network_rates(in.store.folder)
        voltage_tracker = LKD.read_neuron_membrane(in.store.folder)
        adaptation_current_tracker = LKD.read_neuron_membrane(in.store.folder; type="w_adapt")
        adaptive_threshold_tracker = LKD.read_neuron_membrane(in.store.folder; type="adaptive_threshold")
        trackers = voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker
        W_last = w_trace[end][2]
    end
    in.weights = W_last
    return (snn_layer=SNNLayer(in), out=SNNOut(W_last, T, R, trackers, SS_phones, SS_words), weights_trace=w_trace)
end

"""
loads network from path, used to load an existing network in train()
returns SNNOut
"""
function load(folder::String)
    datafile = joinpath(folder, "output.jld2")
    SS_words = LKD.read_network_states(joinpath(folder,"word_states"))
    SS_phones = LKD.read_network_states(joinpath(folder,"phone_states"))
    if isfile(datafile)
        f = jldopen(joinpath(in.store.folder, "output.jld2"), "r")
        W = f["weights"]
        T = f["spikes"]
        R = f["rates"]
        voltage_tracker = f["voltage_tracker"]
        adaptation_current_tracker = f["adaptation_current_tracker"]
        adaptive_threshold_tracker = f["adaptive_threshold_tracker"]
        trackers = voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker
        close(f)
    else
        # "legacy" mode
        W = LKD.read_network_weights(folder)
        T = LKD.read_network_spikes(folder)
        R = LKD.read_network_rates(folder)
        voltage_tracker = LKD.read_neuron_membrane(folder)
        adaptation_current_tracker = LKD.read_neuron_membrane(folder; type="w_adapt")
        adaptive_threshold_tracker = LKD.read_neuron_membrane(folder; type="adaptive_threshold")
        trackers = voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker
        W = W[end][2]
    end
    return SNNOut(W, T, R, trackers, SS_phones, SS_words)
end

"""
inject input from new input layer into a pretrained network layer
"""
function inject(new::tt.InputLayer, old::tt.InputLayer, weights::Matrix{Float64})
    _old = SNNLayer(old)
    _old.store = new.store
    _old.weights = weights
    _old.spikes_dt = new.spikes_dt
    _old.transcriptions_dt = new.transcriptions_dt
    #new.net.simulation_time = old.net.simulation_time # copy simulation time over to be used in the classifier
    _old.net.simulation_time = new.net.simulation_time # update simulation time
    @info "Simulation time is $(_old.net.simulation_time)"
    return _old
end