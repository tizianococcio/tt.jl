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
    trackers:: Union{trackers_voltage, trackers_triplet_basic, trackers_triplet, TrackersT} 
    phone_states::Vector{Any}
    word_states::Vector{Any}
end

function _run(snn::SNNLayer, traces::Bool=false, eSTDP=true, fh=false)
    @assert (!eSTDP && traces) || (eSTDP&&traces) || (!traces && eSTDP) "Excitatory STDP can be turned off only if running with traces."
    if traces
        if eSTDP
            W, T, R, trackers = tt.sim_m(snn.weights, snn.popmembers, snn.spikes_dt, snn.transcriptions_dt, 
                snn.net, snn.store, snn.weights_params, snn.projections, snn.stdp);
        else
            W, T, R, trackers = tt.sim_m_eSTDPoff(snn.weights, snn.popmembers, snn.spikes_dt, snn.transcriptions_dt, 
            snn.net, snn.store, snn.weights_params, snn.projections, snn.stdp);
        end
    elseif fh
        # to test faster homeostatic mechanisms
        W, T, R, trackers = tt.sim_fh(snn.weights, snn.popmembers, snn.spikes_dt, snn.transcriptions_dt, 
        snn.net, snn.store, snn.weights_params, snn.projections, snn.stdp);
    else
        W, T, R, trackers = tt.sim(snn.weights, snn.popmembers, snn.spikes_dt, snn.transcriptions_dt, 
            snn.net, snn.store, snn.weights_params, snn.projections, snn.stdp);
    end
    ws = LKD.read_network_states(joinpath(snn.store.folder,"word_states"));
    ps = LKD.read_network_states(joinpath(snn.store.folder,"phone_states"));
    return SNNOut(W, T, R, trackers, ps, ws)
end

function _run_traces(snn::SNNLayer, full_matrix=false)
    LKD.makefolder(snn.store.folder);
    LKD.cleanfolder(snn.store.folder);
    if !full_matrix
        # only synapses
        tt.sim_mall(snn.weights, snn.popmembers, snn.spikes_dt, snn.transcriptions_dt, 
        snn.net, snn.store, snn.weights_params, snn.projections, snn.stdp);
    else
        #full matrices
        tt.sim_mall_full(snn.weights, snn.popmembers, snn.spikes_dt, snn.transcriptions_dt, 
        snn.net, snn.store, snn.weights_params, snn.projections, snn.stdp);
    end
end

function _run_eSTDPoff(snn::SNNLayer)
    LKD.makefolder(snn.store.folder);
    LKD.cleanfolder(snn.store.folder);
    tt.sim_m_eSTDPoff(snn.weights, snn.popmembers, snn.spikes_dt, snn.transcriptions_dt, 
    snn.net, snn.store, snn.weights_params, snn.projections, snn.stdp);
end

function _run_triplet_barebones(snn::SNNLayer)
    snn.store.folder *= "_bb"
    LKD.makefolder(snn.store.folder);
    LKD.cleanfolder(snn.store.folder);
    tt.sim_bb(snn.weights, snn.popmembers, snn.spikes_dt, snn.transcriptions_dt, 
    snn.net, snn.store, snn.weights_params, snn.projections, snn.stdp);
end

function _run_triplet_deterministic(snn::SNNLayer, dua=true)
    snn.store.folder *= "_det"
    LKD.makefolder(snn.store.folder);
    LKD.cleanfolder(snn.store.folder);
    snn.store.save_states = false
    snn.store.save_network = false
    snn.store.save_weights = false
    tt.sim_det(snn.weights, snn.popmembers, snn.spikes_dt, snn.transcriptions_dt, 
    snn.net, snn.store, snn.weights_params, snn.projections, snn.stdp, dua);
end

function _run_flex_tracks(snn::SNNLayer, ntrack::Int)
    snn.store.save_states = false
    snn.store.save_weights = false
    tt.sim_m_flex(snn.weights, snn.popmembers, snn.spikes_dt, snn.transcriptions_dt, 
    snn.net, snn.store, snn.weights_params, snn.projections, snn.stdp, ntrack);
end
function _run_async(snn::SNNLayer, ch::Channel)
    snn.store.save_states = false
    snn.store.save_network = false
    snn.store.save_weights = false
    W, T, R, trackers = tt.sim_async(snn.weights, snn.popmembers, snn.spikes_dt, snn.transcriptions_dt, 
    snn.net, snn.store, snn.weights_params, snn.projections, snn.stdp, ch);
    return SNNOut(W, T, R, trackers, [], [])    
end

"""
runs the simulation and stores the network states. Overwrite previous data.
`trackers` defines how many neurons to track
fh=faster homeostatic
"""
function train(snn::SNNLayer; overwrite=false, with_traces=false, eSTDP=true, fh=false)

    if isdir(snn.store.folder)
        if overwrite
            LKD.makefolder(snn.store.folder);
            LKD.cleanfolder(snn.store.folder);    
            return _run(snn, with_traces, eSTDP, fh)
        else
            # load trained data and return it
            @info "A trained network already exists. Now loading it. To wipe it and run a new training pass overwrite=true."
            return load(snn.store.folder)
        end
    else
        LKD.makefolder(snn.store.folder);
        LKD.cleanfolder(snn.store.folder);
        return _run(snn, with_traces, eSTDP, fh)
    end

end

function train_with_traces(snn::SNNLayer; overwrite=false, eSTDP=true)
    train(snn; overwrite=overwrite, with_traces=true, eSTDP=eSTDP)
end


"""
runs the simulation on a previously trained network. Data for this network must exist on disk.
transient: if true removes the entire simulation data from disk at simulation end
"""
function test(snn::SNNLayer; trial=0, ntrack = 0, transient=false, fh=false)
    snn.net.learning = false
    if trial == 0
        snn.store.save_weights = false
        # overwrites state in the current folder but keep weights and previous trials
        tt.preparefolder(snn.store.folder)
        if ntrack > 0
            res = _run_flex_tracks(snn, ntrack)
        else
            res = _run(snn, false, true, fh)
        end
    else
        # additional trials: creates subfolders
        if !isdir(snn.store.folder)
            mkdir(snn.store.folder)
        end
        original_folder = snn.store.folder
        snn.store.folder = joinpath(snn.store.folder, "trials", string(trial))
        if !ispath(snn.store.folder)
            tt.LKD.makefolder(snn.store.folder)
            tt.LKD.cleanfolder(snn.store.folder)
            snn.store.save_states=true
            snn.store.save_network=true
            snn.store.save_weights=true
            @info snn.store
            if ntrack > 0
                res = _run_flex_tracks(snn, ntrack)
            else
                res = _run(snn, false, true, fh)
            end
            if transient
                rm(snn.store.folder; recursive=true)
            end
        else
            @info "trial already exists, now loading it."
            res = load(snn.store.folder)
        end
        snn.store.folder = original_folder
    end
    return res
end

function test_with_traces(snn::SNNLayer)
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
    return _run(snn, true)
end

function _loadtrackers(f)
    if haskey(f, "trackers")
        trackers = f["trackers"]
    else
        voltage_tracker = f["voltage_tracker"]
        adaptation_current_tracker = f["adaptation_current_tracker"]
        adaptive_threshold_tracker = f["adaptive_threshold_tracker"]
        trackers = voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker
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
    end
    trackers
end

"""
loads network from disk and updates `in` with the stored weights
"""
function load(in::tt.InputLayer)
    datafile = joinpath(in.store.folder, "output.jld2")
    SS_words = LKD.read_network_states(joinpath(in.store.folder,"word_states"))
    SS_phones = LKD.read_network_states(joinpath(in.store.folder,"phone_states"))
    files = tt.get_weight_files_list(in)
    paths = map(files -> files[2], files)
    
    if isfile(datafile)
        f = jldopen(datafile, "r")
        W_last = f["weights"]
        T = f["spikes"]
        R = f["rates"]
        trackers = _loadtrackers(f)
        close(f)
        w_trace = tt.WeightTrace(length(paths), paths)
    else
        w_trace = tt.WeightTrace(length(paths), paths)
        T = LKD.read_network_spikes(in.store.folder)
        R = LKD.read_network_rates(in.store.folder)
        voltage_tracker = LKD.read_neuron_membrane(in.store.folder)
        adaptation_current_tracker = LKD.read_neuron_membrane(in.store.folder; type="w_adapt")
        adaptive_threshold_tracker = LKD.read_neuron_membrane(in.store.folder; type="adaptive_threshold")
        trackers = voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker
        W_last = w_trace[end]
    end
    in.weights = W_last
    return (snn_layer=SNNLayer(in), out=SNNOut(W_last, T, R, trackers, SS_phones, SS_words), weights_trace=w_trace)
end

function get_weight_traces(in::InputLayer, just_matrices=false)
    @assert isdir(in.store.folder) "Network $(in.id) not found."
    files = tt.get_weight_files_list(in)
    ts = map(files -> files[1], files)
    paths = map(files -> files[2], files)
    w_trace = tt.WeightTrace(length(paths), paths)
    if just_matrices
        w_trace
    else
        zip(ts, w_trace)
    end
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
        f = jldopen(joinpath(folder, "output.jld2"), "r")
        W = f["weights"]
        T = f["spikes"]
        R = f["rates"]
        trackers = _loadtrackers(f)
        close(f)
    else
        # "legacy" mode
        files = tt.get_weight_files_list(folder)
        paths = map(files -> files[2], files)        
        W = tt.WeightTrace(length(paths), paths)
        T = LKD.read_network_spikes(folder)
        R = LKD.read_network_rates(folder)
        voltage_tracker = LKD.read_neuron_membrane(folder)
        adaptation_current_tracker = LKD.read_neuron_membrane(folder; type="w_adapt")
        adaptive_threshold_tracker = LKD.read_neuron_membrane(folder; type="adaptive_threshold")
        trackers = voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker
        W = W[end]
    end
    return SNNOut(W, T, R, trackers, SS_phones, SS_words)
end

function delete_sim(il::tt.InputLayer)
    tt.delete(il)
    folder = joinpath(tt.simsdir(), il.id);
    if ispath(folder)
        rm(folder, recursive=true)
    end
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