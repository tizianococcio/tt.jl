using Statistics
using tt
using JLD2
using LKD
using Plots

# Same as all the other "A" experiments, but does not use a Poisson input, it uses the TIMIT input
using Distributions

function is_neuron_in_assembly(neuron_id, popm)
    for i in 1:size(popm, 2)
        res = findall(x -> x == neuron_id, popm[(popm[:,i] .> -1), i])
        if length(res) > 0
            return true
        end
    end
    return false
end

function make_plot(volt, adaptTh, adaptI, title)
    Plots.plot(volt, label="Membrane potential", linestyle=:solid, color=:black, linewidth=3, linealpha=0.4);
    Plots.plot!(adaptTh, label="Adaptive threshold", linestyle=:dash, color=:black, linewidth=1);
    Plots.plot!(adaptI, label="Adaptation current", linestyle=:dot, color=:black, linewidth=1);
    return Plots.plot!(legend=:none, xlabel="Simulation time steps", title=title, tickfontsize=5, labelfontsize=5, titlefont=7);
end


conf = tt.load_conf()
t_folder_name = "experimentA2_trip"
path_dataset = conf["dataset_path"]
path_storage = conf["training_storage_path"]
path_exp = joinpath(conf["experiments"], "ExperimentA2")
t_folder_train = joinpath(path_storage, t_folder_name)
weights_params = LKD.WeightParams(Ne=200, Ni=50)
store = LKD.StoreParams(folder = t_folder_train, 
    save_states=false, save_network=true, save_weights=true, save_timestep=1000);
LKD.makefolder(store.folder);
LKD.cleanfolder(store.folder);

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
    
shuffled_ids = tt.SpikeTimit.mix_inputs(length(word_inputs.durations), params.repetitions)
spikes, transcriptions = tt.SpikeTimit.get_ordered_spikes(word_inputs, ids=shuffled_ids, silence_time=params.silence_time, shift=params.shift_input, repetitions=params.repetitions);
transcriptions_dt = tt.SpikeTimit.transcriptions_dt(transcriptions);
spikes_dt = tt.SpikeTimit.ft_dt(spikes)
projections = LKD.ProjectionParams(npop = tt.SpikeTimit.get_encoding_dimension(word_inputs.spikes));
W, popmembers = LKD.create_network(weights_params, projections);
w_init = copy(W)
if !is_neuron_in_assembly(1, popmembers)
    print("Neuron not present in any assembly, rerun create_network().")
end
net = LKD.NetParams(learning=true, simulation_time=ceil(transcriptions.phones.intervals[end][end]*1000));
W, T, R, trackers_trip = tt.sim_m(W, popmembers, spikes_dt, transcriptions_dt, net, store, weights_params, projections, tt.TripletSTDP());

# Plot how many neurons are targeted for each encoding neuron
bar(sum(popmembers[:,:] .> -1, dims=1)', xlabel="encoding neuron", ylabel="num. of targeted network neurons")

# Plot for triplet stdp
volt, adaptI, adaptTh, r1, o1, r2, o2, weight_trace = trackers_trip
plot(weight_trace[1][10000:end,:], leg=:none, color=:black, alpha=0.3, title="Triplet: synapses 1-to-all")

# Voltage
v_folder_name = "experimentA2_volt"
path_dataset = conf["dataset_path"]
path_storage = conf["training_storage_path"]
v_folder_train = joinpath(path_storage, v_folder_name)
store = LKD.StoreParams(folder = v_folder_train, 
    save_states=false, save_network=true, save_weights=true, save_timestep=1000);
LKD.makefolder(store.folder);
LKD.cleanfolder(store.folder);
#W, popmembers = LKD.create_network(weights_params, projections);
W = copy(w_init)
net = LKD.NetParams(learning=true, simulation_time=ceil(transcriptions.phones.intervals[end][end]*1000));

## For Voltage-stdp (I expect to see basically the same plot)
_, v_ft, v_fr, trackers_volt = tt.sim_m(W, popmembers, spikes_dt, transcriptions_dt, 
net, store, weights_params, projections, tt.VoltageSTDP());
using Dates

jldopen(joinpath(path_exp, "data$(today()).jld2"), "w") do file
    file["triplet"] = trackers_trip
    file["voltage"] = trackers_volt
end





## Only reading data from here onwards
conf = tt.load_conf()
folder = joinpath(conf["training_storage_path"], "experimentA2_trip");
t_ft = LKD.read_network_spikes(folder)
t_fr = LKD.read_network_rates(folder)

folder = joinpath(conf["training_storage_path"], "experimentA2_volt");
v_ft = LKD.read_network_spikes(folder)
v_fr = LKD.read_network_rates(folder)

fromdisk = load(joinpath(path_exp, "data$(today()).jld2"))
trackers_trip = fromdisk["triplet"]
trackers_volt = fromdisk["voltage"]

t_volt, t_adaptI, t_adaptTh, r1, o1, r2, o2, t_weight_trace = trackers_trip
v_volt, v_adaptI, v_adaptTh, u_tr, v_tr, v_weight_trace = trackers_volt

t_ft = convert(Vector{Vector{Float64}}, t_ft)
v_ft = convert(Vector{Vector{Float64}}, v_ft)



using CairoMakie
f = Figure(resolution = (1000, 600))
ga = f[1:3, 1] = GridLayout()
gb = f[1:3, 2] = GridLayout()

ax1 = Axis(ga[1,1], title="Triplet STDP", ylabel="Neuron", ytickalign=1)
x,y = tt.SpikeTimit.get_raster_data(t_ft)
CairoMakie.scatter!(x, y, color=:black, markersize=1)
hidedecorations!(ax1; grid=true, ticklabels = false, ticks = false, label=false)
hidexdecorations!(ax1);

ax2 = Axis(ga[2,1], ylabel="Weight", ytickalign=1);
# post
data_matrix = t_weight_trace[2][10000:end,:]';
post2 = CairoMakie.series!(data_matrix, solid_color=(:orange, 0.3), strokewidth=0, linewidth=1, markersize=0);
# pre
data_matrix = t_weight_trace[1][10000:end,:]';
pre2 = CairoMakie.series!(data_matrix, solid_color=(:black, 0.3), strokewidth=0, linewidth=1, markersize=0);
#axislegend(position = :lt, labelsize=12, framevisible = false)
axislegend(ax2, [pre2, post2], ["Pre", "Post"], position = :lt, orientation = :horizontal, framevisible = false, labelsize=12);
hidedecorations!(ax2; grid=true, ticklabels = false, ticks = false, label=false);
hidexdecorations!(ax2);

ax3 = Axis(ga[3,1], xlabel="Simulation timestep", ylabel="mA", xtickalign=1, ytickalign=1, height=150);
CairoMakie.lines!(t_adaptI, color=(:black, 1), label="Adapt. current");
CairoMakie.lines!(t_fr[1,:], color=(:black, 0.4), label="Firing rate");
axislegend(position = :rb, labelsize=12, framevisible = false)
hidedecorations!(ax3; grid=true, ticklabels = false, ticks = false, label=false);

ax4 = Axis(gb[1,1], title="Voltage STDP");
x,y = tt.SpikeTimit.get_raster_data(v_ft);
CairoMakie.scatter!(x, y, color=:black, markersize=1);
hidedecorations!(ax4; grid=true, ticklabels = false, ticks = false, label=false);
hidexdecorations!(ax4);
hideydecorations!(ax4);
linkyaxes!(ax1, ax4);

ax5 = Axis(gb[2,1]);
# post
data_matrix = v_weight_trace[2][10000:end,:]';
post5 = CairoMakie.series!(data_matrix, solid_color=(:orange, 0.3), strokewidth=0, linewidth=1, markersize=0);
# pre
data_matrix = v_weight_trace[1][10000:end,:]';
pre5 = CairoMakie.series!(data_matrix, solid_color=(:black, 0.3), strokewidth=0, linewidth=1, markersize=0);
#axislegend(position = :lt, labelsize=12, framevisible = false)
axislegend(ax5, [pre5, post5], ["Pre", "Post"], position = :lt, orientation = :horizontal, framevisible = false, labelsize=12);
hidedecorations!(ax5; grid=true, ticklabels = false, ticks = false, label=false);
hidexdecorations!(ax5);
hideydecorations!(ax5);

ax6 = Axis(gb[3,1], xlabel="Simulation timestep", xtickalign=1, height=150);
CairoMakie.lines!(v_adaptI, color=(:black, 1), label="Adapt. current");
CairoMakie.lines!(v_fr[1,:], color=(:black, 0.4), label="Firing rate");
axislegend(position = :rb, labelsize=12, framevisible = false)
hidedecorations!(ax6; grid=true, ticklabels = false, ticks = false, label=false);
hideydecorations!(ax6);

linkyaxes!(ax2, ax5);
linkyaxes!(ax3, ax6);
ax3.xticks = 0:30000:60000
ax6.xticks = 0:30000:60000
colgap!(ga, 10)
colgap!(gb, 10)
rowgap!(ga, 10)
rowgap!(gb, 10)
f
save("ExperimentA2.pdf", f, pt_per_unit = 2)


# Detail of weight traces
f_detail = Figure()
g = f_detail[1:2, 1] = GridLayout()
axtop = Axis(g[1, 1], title="TripletSTDP")
axbottom = Axis(g[2, 1], title="VoltageSTDP")

# triplet
# post
data_matrix = t_weight_trace[2][10000:end,1:5]';
aa = CairoMakie.series!(axtop, data_matrix, solid_color=(:orange, 0.5), strokewidth=0, linewidth=1, markersize=0);
# pre
data_matrix = t_weight_trace[1][10000:end,1:5]';
bb = CairoMakie.series!(axtop, data_matrix, solid_color=(:black, 0.5), strokewidth=0, linewidth=1, markersize=0);
axislegend(axtop, [bb, aa], ["Pre", "Post"], position = :lt, orientation = :horizontal, framevisible = false, labelsize=12)
hidedecorations!(axtop, grid=true, ticklabels = false, ticks = false, label=false);

# voltage
# post
data_matrix = v_weight_trace[2][10000:end,1:5]';
aa = CairoMakie.series!(axbottom, data_matrix, solid_color=(:orange, 0.5), strokewidth=0, linewidth=1, markersize=0);
# pre
data_matrix = v_weight_trace[1][10000:end,1:5]';
bb = CairoMakie.series!(axbottom, data_matrix, solid_color=(:black, 0.5), strokewidth=0, linewidth=1, markersize=0);
#axislegend(position = :lt, labelsize=12, framevisible = false)
axislegend(axbottom, [bb, aa], ["Pre", "Post"], position = :lt, orientation = :horizontal, framevisible = false, labelsize=12)
hidedecorations!(axbottom, grid=true, ticklabels = false, ticks = false, label=false);
f_detail
save("ExperimentA2_detail.pdf", f_detail, pt_per_unit = 2)
#current_figure()


# histograms of weights taken from savepoints
W_trip = LKD.read_network_weights(t_folder_train)
Ne = 200
n = length(W_trip)

hdata = [filter(x->x!=0, W_trip[i][2][1:Ne,1:Ne][:]) for i in 1:n]
W_volt = LKD.read_network_weights(v_folder_train)
hdata_v = [filter(x->x!=0, W_volt[i][2][1:Ne,1:Ne][:]) for i in 1:n]

p = plot(histogram(hdata[15], color=:red, xlabel="Weight", ylabel="Count", title="Triplet ($(1/10))"), 
        histogram(hdata_v[15], color=:red, yticks=false, xlabel="Weight", title="Voltage"), 
    leg=false, xlims=[1,6], ylims=[0,800], grid=false, linewidth=0, framestyle=:box)

anim = Animation()
anim = @animate for i ∈ 1:n
    plot(
        histogram(hdata[i], color=:red, xlabel="Weight", ylabel="Count", title="Triplet ($(i/10))"), 
        histogram(hdata_v[i], color=:red, yticks=false, xlabel="Weight", title="Voltage ($(i/10))"), 
        leg=false, 
        xlims=[1,6], 
        ylims=[0,800], 
        grid=false, 
        linewidth=0, 
        framestyle=:box
    )    
end
gif(anim, "anim_weights.gif", fps=10)


# an attempt at having a scatter plot of the input with a bar that scrolls through it
duration, spikes, labels = word_inputs
x,y=tt.SpikeTimit.get_raster_data(spikes[1])
anim = @animate for i in 0:0.001:maximum(x)
    scatter(x,y)
    vline!([i])
end
gif(anim, "scat.gif", fps = 10)
