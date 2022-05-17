using YAML
using Plots
using tt
using JLD2
using DataFrames

conf = YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"));
path_dataset = conf["dataset_path"]
path_storage = conf["training_storage_path"]
path_experiments = conf["experiments"]

folder = mkdir(joinpath(path_experiments, "ExperimentF0"));
folder_df = joinpath(folder, "df_results.jld2");

params = tt.LKD.InputParams(
    dialects=[1,2,3,4], 
    samples=8, 
    gender=['m', 'f'], 
    words=["the", "a", "water", "greasy"], 
    repetitions=6, 
    shift_input=2, 
    encoding="cochlea70"
);
weights_params = tt.LKD.WeightParams()

input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.TripletSTDP());
snn = tt.SNNLayer(input);
global snn_out;
for i in 1:2
    println(">Training iteration n.$i")
    snn_out = tt.train(snn);
    snn.weights = snn_out.weights
end

jldopen(joinpath(folder, "data.jld2"), "w") do file
    file["triplet"] = snn_out.trackers
end

if isfile(folder_df)
    df_class = JLD2.load(folder_df, "dataframe_classifier");
else
    df_class = DataFrame(InputParam=tt.LKD.InputParams[], score=Float64[], feature_type=String[], learning_rule=String[])
end

triplet = tt.load(input);

classifier = tt.ClassificationLayer(snn_out)
score, prms = tt.on_spikes(classifier, input)
push!(df_class, Dict(:InputParam => params, :score => score, :feature_type => "spikes", :learning_rule => "triplet"))


# present novel stimulus
params = tt.LKD.InputParams(
    dialects=[5], 
    samples=8, 
    gender=['m', 'f'], 
    words=["water"], 
    repetitions=3, 
    shift_input=2, 
    encoding="cochlea70"
);

new_input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.TripletSTDP());
@assert triplet.snn_layer.spikes_dt != new_input.spikes_dt
# inject new stimulus into old network
tt.inject(new_input, triplet.snn_layer)
@assert triplet.snn_layer.spikes_dt == new_input.spikes_dt
snn_out = tt.test(triplet.snn_layer);
classifier = tt.ClassificationLayer(snn_out);
score, _ = tt.on_spikes(classifier, new_input);


##############
# VoltageSTDP
params = tt.LKD.InputParams(
    dialects=[1,2,3,4], 
    samples=8, 
    gender=['m', 'f'], 
    words=["the", "a", "water", "greasy"], 
    repetitions=6, 
    shift_input=2, 
    encoding="cochlea70"
);
input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP());
snn = tt.SNNLayer(input);
global snn_out;

for i in 1:2
    println(">Training iteration n.$i")
    snn_out = tt.train(snn);
    snn.weights = snn_out.weights
end
classifier = tt.ClassificationLayer(snn_out);
score, _ = tt.on_spikes(classifier, input)
push!(df_class, Dict(:InputParam => params, :score => score, :feature_type => "spikes", :learning_rule => "voltage"))



jldopen(folder_df, "w") do file
    file["dataframe_classifier"] = df_class
end;
