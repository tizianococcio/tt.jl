using Statistics
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
    Plots.plot!(adaptI, label="Adaptation current", linestyle=:dot, color=:black, linewidth=2);
    return Plots.plot!(legend=:none, xlabel="Simulation time steps", title=title, ylims=[-60,20], tickfontsize=5, labelfontsize=5, titlefont=7);
end



simtime = 10000
dt = 0.1f0
firing_rate_hz = 0.4
input = PoissonInput(firing_rate_hz, simtime, dt; neurons=2)
ft1 = findall(x -> x != 0, input[1,:])
ft2 = findall(x -> x != 0, input[2,:])
fts = vcat(ft1,ft2)
sort!(fts)
neurons = Vector{Vector{Int64}}()
for ft_i in eachindex(fts)
    push!(neurons, [])
    if fts[ft_i] in ft1
        push!(neurons[ft_i], 1)
    end
    if fts[ft_i] in ft2
        push!(neurons[ft_i], 2)
    end
end
spikes_dt = (ft = fts, neurons = neurons);

folder_name = "experimentA"
conf = tt.load_conf()
path_dataset = conf["dataset_path"]
path_storage = conf["training_storage_path"]
folder = joinpath(path_storage, folder_name)
weights_params = LKD.WeightParams(Ne=2, Ni=0)
projections = LKD.ProjectionParams(npop = 2);
W, popmembers = LKD.create_network(weights_params, projections);
net = LKD.NetParams(learning=true);
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

_, _, _, trackers_trip = tt.sim(W, popmembers, spikes_dt, transcriptions_dt, net, store, weights_params, projections, tt.TripletSTDP());
W_r= LKD.read_network_weights(folder)


W, popmembers = LKD.create_network(weights_params, projections);
net = LKD.NetParams(learning=true);
store = LKD.StoreParams(folder = joinpath(path_storage, folder_name), save_states=false, save_network=true, save_weights=true);

## For Voltage-stdp (I expect to see basically the same plot)
_, _, _, trackers_volt = tt.sim(W, popmembers, spikes_dt, transcriptions_dt, 
net, store, weights_params, projections, tt.VoltageSTDP());


jldopen(joinpath(folder, "data.jld2"), "w") do file
    file["triplet"] = trackers_trip
    file["voltage"] = trackers_volt
end

fromdisk = load(joinpath(folder, "data.jld2"))
trackers_trip = fromdisk["triplet"]
trackers_volt = fromdisk["voltage"]

volt, adaptI, adaptTh = trackers_trip
ptrip = make_plot(volt, adaptTh, adaptI, "Triplet STDP");

volt, adaptI, adaptTh = trackers_volt
pvolt = make_plot(volt, adaptTh, adaptI, "Voltage STDP");

p = Plots.plot(ptrip, pvolt, ylabel="mV", size=(800, 400), legend=:outertop)
Plots.savefig(p, "ExperimentA.pdf")


# work in progress: figure with plotly
# https://plotly.com/julia/line-charts/
using PlotlyJS

layout = Layout(plot_bgcolor="white",
    xaxis=attr(
        showline=true,
        showgrid=false,
        showticklabels=true,
        linecolor="black",
        ticks="outside"
    ),
    yaxis=attr(
        showline=true,
        showgrid=false,
        showticklabels=true,
        linecolor="black",
        ticks="outside"
    )
)
p1 = plot(scatter(x=1:3, y=4:6, marker=attr(color="black", dash="dash")), layout)
p2 = plot(scatter(x=20:40, y=50:70), layout)
p = [p1 p2]
relayout!(p, title_text="Side by side layout (1 x 2)", layout=layout)
p
savefig(p, "test.pdf")