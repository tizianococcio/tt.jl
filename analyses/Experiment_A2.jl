using Statistics
using YAML
using tt
using JLD2
using SpikeTimit
using LKD
using Plots

# Same as all the other "A" experiments, but does not use a Poisson input, it uses the TIMIT input
using Distributions


function make_plot(volt, adaptTh, adaptI, title)
    Plots.plot(volt, label="Membrane potential", linestyle=:solid, color=:black, linewidth=3, linealpha=0.4);
    Plots.plot!(adaptTh, label="Adaptive threshold", linestyle=:dash, color=:black, linewidth=1);
    Plots.plot!(adaptI, label="Adaptation current", linestyle=:dot, color=:black, linewidth=1);
    return Plots.plot!(legend=:none, xlabel="Simulation time steps", title=title, tickfontsize=5, labelfontsize=5, titlefont=7);
end


folder_name = "experimentA2_trip"
conf = YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
path_dataset = conf["dataset_path"]
path_storage = conf["training_storage_path"]
folder = joinpath(path_storage, folder_name)
weights_params = LKD.WeightParams(Ne=200, Ni=50)
store = LKD.StoreParams(folder = joinpath(path_storage, folder_name), 
    save_states=false, save_network=true, save_weights=true, save_timestep=1000);
folder = LKD.makefolder(store.folder);
folder = LKD.cleanfolder(store.folder);

# dummy transcriptions data to make the simulation run
params = tt.LKD.InputParams(
    dialects=[1,2,3], 
    samples=20, 
    gender=['m'], 
    words=["music"], 
    repetitions=6, 
    shift_input=2, 
    encoding="cochlea70"
)
df = tt.load_dataset(path_dataset, params)
word_inputs = tt.SpikeTimit.select_words(
    df, 
    samples=params.samples, 
    params.words, 
    encoding=params.encoding);
    
shuffled_ids = SpikeTimit.mix_inputs(length(word_inputs.durations), params.repetitions)
spikes, transcriptions = SpikeTimit.get_ordered_spikes(word_inputs, ids=shuffled_ids, silence_time=params.silence_time, shift=params.shift_input, repetitions=params.repetitions);
transcriptions_dt = SpikeTimit.transcriptions_dt(transcriptions);
spikes_dt = SpikeTimit.ft_dt(spikes)
projections = LKD.ProjectionParams(npop = SpikeTimit.get_encoding_dimension(word_inputs.spikes));
W, popmembers = LKD.create_network(weights_params, projections);
net = LKD.NetParams(learning=true, simulation_time=ceil(transcriptions.phones.intervals[end][end]*1000));
W, T, R, trackers_trip = tt.sim_m(W, popmembers, spikes_dt, transcriptions_dt, net, store, weights_params, projections, tt.TripletSTDP());


# Plot for triplet stdp
volt, adaptI, adaptTh, r1, o1, r2, o2, weight_trace = trackers_trip
plot(weight_trace[10000:end,:], leg=:none, color=:black, alpha=0.3, title="Triplet: synapses 1-to-all")

# Voltage
folder_name = "experimentA2_volt"
conf = YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
path_dataset = conf["dataset_path"]
path_storage = conf["training_storage_path"]
folder = joinpath(path_storage, folder_name)
weights_params = LKD.WeightParams(Ne=200, Ni=50)
store = LKD.StoreParams(folder = joinpath(path_storage, folder_name), 
    save_states=false, save_network=true, save_weights=true);
folder = LKD.makefolder(store.folder);
folder = LKD.cleanfolder(store.folder);
W, popmembers = LKD.create_network(weights_params, projections);
net = LKD.NetParams(learning=true, simulation_time=ceil(transcriptions.phones.intervals[end][end]*1000));
store = LKD.StoreParams(folder = joinpath(path_storage, folder_name), save_states=false, save_network=true, save_weights=true);

## For Voltage-stdp (I expect to see basically the same plot)
_, v_ft, v_fr, trackers_volt = tt.sim_m(W, popmembers, spikes_dt, transcriptions_dt, 
net, store, weights_params, projections, tt.VoltageSTDP());

jldopen(joinpath(folder, "data.jld2"), "w") do file
    file["triplet"] = trackers_trip
    file["voltage"] = trackers_volt
end





## Only reading data from here onwards
conf = YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
folder = joinpath(conf["training_storage_path"], "experimentA2_trip");
t_ft = LKD.read_network_spikes(folder)
t_fr = LKD.read_network_rates(folder)

folder = joinpath(conf["training_storage_path"], "experimentA2_volt");
v_ft = LKD.read_network_spikes(folder)
v_fr = LKD.read_network_rates(folder)

fromdisk = load(joinpath(folder, "data.jld2"))
trackers_trip = fromdisk["triplet"]
trackers_volt = fromdisk["voltage"]



# plot grid
t_ft = convert(Vector{Vector{Float64}}, t_ft)
v_ft = convert(Vector{Vector{Float64}}, v_ft)
psr_t = scatter(SpikeTimit.get_raster_data(t_ft), m=(1, :black, stroke(0)), leg = :none, yticks=false, xtickfontsize=5, labelfontsize=5, grid=false)
psr_v = scatter(SpikeTimit.get_raster_data(v_ft), m=(1, :black, stroke(0)), leg = :none, yticks=false, xtickfontsize=5, labelfontsize=5, grid=false)
t_volt, t_adaptI, t_adaptTh, r1, o1, r2, o2, t_weight_trace = trackers_trip
v_volt, v_adaptI, v_adaptTh, u_tr, v_tr, v_weight_tracker = trackers_volt
pw_t = plot(t_weight_trace[10000:end,:], leg=:none, color=:black, alpha=0.3, title="Triplet: synapses 1-to-all")
pw_v = plot(v_weight_tracker[10000:end,:], leg=:none, color=:black, alpha=0.3, title="Voltage: synapses 1-to-all")
pmb_t = plot(t_adaptI, color=:black);
pmb_v = plot(v_adaptI, color=:black);
lyt = @layout [a b; a1 b1; a2 b2]
plot(reduce(hcat, [pw_t, pw_v, psr_t, psr_v, pmb_t, pmb_v])..., layout=lyt, size=(1000,600))


plot(t_volt, size=(1000,300))
plot(t_adaptI)
plot(t_adaptTh, xlims=[10000,20000])
plot(r1, size=(1000,300))

using CairoMakie

f = Figure(resolution = (1000, 600))
ga = f[1:3, 1] = GridLayout()
gb = f[1:3, 2] = GridLayout()

ax = Axis(ga[1,1])
x,y = SpikeTimit.get_raster_data(t_ft)
CairoMakie.scatter!(x, y, color=:black, markersize=1)
hidedecorations!(ax; grid=true, ticklabels = false, ticks = false, label=false)


ax = Axis(ga[2,1])
data_matrix = t_weight_trace[10000:end,:]'
CairoMakie.series!(data_matrix, solid_color=(:black, 0.3), strokewidth=0, linewidth=1, markersize=0)
hidedecorations!(ax; grid=true, ticklabels = false, ticks = false, label=false)

ax = Axis(ga[3,1])
CairoMakie.lines!(t_adaptI)
hidedecorations!(ax; grid=true, ticklabels = false, ticks = false, label=false)

f

fig, ax, sp = CairoMakie.series(data_matrix, solid_color=(:black, 0.3), strokewidth=0, linewidth=1, markersize=0)
fig