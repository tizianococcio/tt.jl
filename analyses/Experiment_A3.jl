# Same as all the other "A" experiments, uses a Poisson input, with 250 neurons
using Statistics
using YAML
using tt
using JLD2
using LKD
using Plots

using Distributions

function _PoissonInput(Hz_rate::Real, interval::Int64, dt::Float32)
    λ = 1000/Hz_rate
	spikes = falses(round(Int,interval/dt))
	t = 1
	while t < interval/dt
		Δ = rand(Exponential(λ/dt))
		t += Δ
		if t < interval/dt
			spikes[round(Int,t)] = true
		end
	end
	return spikes
end

function PoissonInput(Hz_rate::Real, interval::Int64, dt::Float32; neurons::Int64=1)
	spikes = falses(neurons, round(Int,interval/dt))
	for n in 1:neurons
		spikes[n,:] .= _PoissonInput(Hz_rate::Real, interval::Int64, dt::Float32)
	end
	return spikes
end

function make_plot(volt, adaptTh, adaptI, title)
    Plots.plot(volt, label="Membrane potential", linestyle=:solid, color=:black, linewidth=3, linealpha=0.4);
    Plots.plot!(adaptTh, label="Adaptive threshold", linestyle=:dash, color=:black, linewidth=1);
    Plots.plot!(adaptI, label="Adaptation current", linestyle=:dot, color=:black, linewidth=1);
    return Plots.plot!(legend=:none, xlabel="Simulation time steps", title=title, tickfontsize=5, labelfontsize=5, titlefont=7);
end



simtime = 10000
dt = 0.1f0
firing_rate_hz = 0.4
input = PoissonInput(firing_rate_hz, simtime, dt; neurons=200)
fts = []
for i in 1:200
    all_neurons_at_i = findall(x -> x != 0, input[i,:])
    push!(fts, all_neurons_at_i)
end
fts = vcat(fts...)
neurons = Vector{Vector{Int64}}(undef, length(fts))
sort!(fts)
for i in 1:200
    timestamps = findall(x -> x != 0, input[i,:])
    for time in timestamps
        id = findfirst(x->x == time, fts)
        if !isassigned(neurons, id)
            neurons[id] = Vector{Int64}()
        end
        push!(neurons[id], i)
    end
end

spikes_dt = (ft = fts, neurons = neurons);

folder_name = "experimentA3"
conf = YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
path_dataset = conf["dataset_path"]
path_storage = conf["training_storage_path"]
folder = joinpath(path_storage, folder_name)
weights_params = LKD.WeightParams(Ne=200, Ni=50)
projections = LKD.ProjectionParams(npop = 10);
W, popmembers = LKD.create_network(weights_params, projections);
net = LKD.NetParams(learning=true, simulation_time=simtime);
#net = LKD.NetParams(learning=true);
store = LKD.StoreParams(folder = joinpath(path_storage, folder_name), 
    save_states=false, save_network=true, save_weights=true);
folder = LKD.makefolder(store.folder);
folder = LKD.cleanfolder(store.folder);

# dummy transcriptions data to make the simulation run
params = tt.LKD.InputParams(
    dialects=[1], 
    samples=1, 
    gender=['m'], 
    words=["music"], 
    repetitions=1, 
    shift_input=2, 
    encoding="cochlea70"
)
df = tt.load_dataset(path_dataset, params)
word_inputs = tt.SpikeTimit.select_words(
    df, 
    samples=params.samples, 
    params.words, 
    encoding=params.encoding);
_, transcriptions = tt.SpikeTimit.get_ordered_spikes(word_inputs, ids=[1],silence_time=params.silence_time,shift=params.shift_input);
transcriptions_dt = tt.SpikeTimit.transcriptions_dt(transcriptions);

_, _, _, trackers_trip = tt.sim_m(W, popmembers, spikes_dt, transcriptions_dt, net, store, weights_params, projections, tt.TripletSTDP());
W_r= LKD.read_network_weights(folder)

W = copy(initial_W)
store = LKD.StoreParams(folder = joinpath(path_storage, folder_name), save_states=false, save_network=true, save_weights=true);
folder = LKD.cleanfolder(store.folder);

## For Voltage-stdp (I expect to see basically the same plot)
_, _, _, trackers_volt = tt.sim_m(W, popmembers, spikes_dt, transcriptions_dt, 
net, store, weights_params, projections, tt.VoltageSTDP());

jldopen(joinpath(folder, "data.jld2"), "w") do file
    file["triplet"] = trackers_trip
    file["voltage"] = trackers_volt
end

volt, adaptI, adaptTh, r1, o1, r2, o2, weight_trace = trackers_trip
volt, adaptI, adaptTh, u_tr, v_tr = trackers_volt

# fromdisk = load(joinpath(folder, "data.jld2"))
# trackers_trip = fromdisk["triplet"]
# trackers_volt = fromdisk["voltage"]
