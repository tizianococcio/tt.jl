# Evolution of weights

using YAML
using Plots
using tt
using JLD2

conf = YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
path_dataset = conf["dataset_path"]
path_storage = conf["training_storage_path"]

params = tt.LKD.InputParams(
    dialects=[1], 
    samples=10, 
    gender=['m'], 
    words=["greasy"], 
    repetitions=3, 
    shift_input=2, 
    encoding="cochlea70"
);
df = tt.load_dataset(path_dataset, params)
word_inputs = tt.SpikeTimit.select_words(
    df, 
    samples=params.samples, 
    params.words, 
    encoding=params.encoding
);
        
# getting the spikes here
duration, spikes, labels = word_inputs
        
weights_params = tt.LKD.WeightParams()
input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.TripletSTDP())
input.store.save_timestep = 15000
snn = tt.SNNLayer(input)
out = tt.train(snn);

input.store.folder
ws = tt.LKD.read_network_weights(input.store.folder);
Ne = 40
heatmap(ws[1][2][1:Ne,1:Ne])
heatmap(ws[2][2][1:Ne,1:Ne])
heatmap(ws[3][2][1:Ne,1:Ne])
heatmap(ws[4][2][1:Ne,1:Ne])
plot!("triplet")
voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker, r, o = out.trackers
range = 10000:20000
dtcs = hcat(r, o)
plot(dtcs[range,:], label=["pre" "post"])
plot!(out.firing_rates[1,:][range])
out.firing_times
plot(out.firing_times)
plot(voltage_tracker)


## VoltageSTDP
weights_params = tt.LKD.WeightParams()
input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP())
input.store.save_timestep = 1000
snn = tt.SNNLayer(input)
out = tt.train(snn);
input.store.folder
ws = tt.LKD.read_network_weights(input.store.folder);
ws

heatmap(ws[1][2][1:Ne,1:Ne])
heatmap(ws[2][2][1:Ne,1:Ne])
heatmap(ws[3][2][1:Ne,1:Ne])
heatmap(ws[4][2][1:Ne,1:Ne])