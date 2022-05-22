using YAML
using Plots
using tt
using JLD2
using DataFrames
using Dates

conf = tt.load_conf()
path_dataset = conf["dataset_path"]
path_storage = conf["training_storage_path"]
path_experiments = conf["experiments"]

# prep folder for experiment data
folder = joinpath(path_experiments, "ExperimentFp0")
if !isdir(folder)
    mkdir(folder);
end
folder_df = joinpath(folder, "results.jld2"); # dataframe with results

# select input
data_df = tt.get_timit_train_dataframe(path_dataset)
[(length(data_df[i,:].sentence), join(data_df[i,:].sentence, " ")) for i in 1:20]
join(data_df[19,:].sentence, " ")
join(data_df[57,:].sentence, " ")
join(data_df[64,:].sentence, " ")
join(data_df[19,:].sentence, " ")
input_words = data_df[19,:].sentence
data_df[19,:]

function dfget(folder)
    if isfile(folder)
        df_class = JLD2.load(folder_df, "df_class");
    else
        df_class = DataFrame(
            dialects=Vector{Int64}[],
            samples=Int64[],
            gender=Vector{Char}[],
            words=Vector{String}[],
            repetitions=Int64[],
            shift_input=Float64[],
            encoding=String[],
            score=Float64[], 
            feature_type=String[], 
            learning_rule=String[]
        );
    end
    return df_class
end
function dfsave(df, folder)
    jldopen(folder, "w") do file
        file["df_class"] = df
    end;
end
function dfadd(df, params::LKD.InputParams, score::Float64, feature_type::String, learning_rule::String)
    push!(df, Dict(
        :dialects => params.dialects,
        :samples => params.samples,
        :gender => params.gender,
        :words => params.words,
        :repetitions => params.repetitions,
        :shift_input => params.shift_input,
        :encoding => params.encoding,
        :score => score,
        :feature_type => feature_type,
        :learning_rule => learning_rule
    ))
end



##############
# VoltageSTDP
weights_params = tt.LKD.WeightParams()
params = tt.LKD.InputParams(
    dialects=[2], 
    samples=1, 
    gender=['f'], 
    words=input_words, 
    repetitions=10, # epochs 
    shift_input=2, 
    encoding="bae"
);
input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP());
snn = tt.SNNLayer(input);
snn_out = tt.train(snn);
snn.transcriptions_dt.words

# presenting network with exactly the same input
test_output = tt.test(snn)
classifier = tt.ClassificationLayer(test_output);

# classify on spikes
score, _ = tt.on_spikes(classifier, input)
dfadd(df, params, score, "spikes", "voltage")
df

# classify on words
score, _ = tt.on_words(classifier)
dfadd(df, params, score, "words", "voltage")
df


dfsave(df, folder_df)

# present network with an individual word
word = input_words[3:3]
params = tt.LKD.InputParams(
    dialects=[2], 
    samples=1, 
    gender=['f'], 
    words=word, 
    repetitions=1, # epochs 
    shift_input=2, 
    encoding="bae"
);

test_input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP());

input.popmembers == test_input.popmembers
test_input
test_input.weights = copy(snn_out.weights)
test_input.popmembers = copy(input.popmembers)
test_snn = tt.SNNLayer(test_input)
test_snn.weights == snn_out.weights
test_snn_out = tt.test(test_snn)

test_snn_out
classifier = tt.ClassificationLayer(test_output);
classifier.word_states
feats, n_neurons, labels = LKD.states_to_features(classifier.word_states)
feats
n_neurons
labels
scores = LKD.MultiLogReg(feats, labels)



score, _ = tt.on_words(classifier)

a



# further training passes
snn.weights = copy(snn_out.weights)
snn_out1 = tt.train(snn)

snn_out.weights
snn.weights
params



# present network with each word individually
word = input_words[1:1]
params = tt.LKD.InputParams(
    dialects=[2], 
    samples=1, 
    gender=['f'], 
    words=word, 
    repetitions=1, # epochs 
    shift_input=2, 
    encoding="bae"
);
println("presenting '$word'");
test_input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP());
test_snn = tt.inject(test_input, snn)
test_output = tt.test(test_snn)
classifier = tt.ClassificationLayer(test_output);
score, _ = tt.on_spikes(classifier, test_input)
interval = 100:100.:test_input.net.simulation_time
spike_feats, spike_labels = LKD.spikes_to_features(classifier.firing_times, test_input.transcriptions.phones, interval)
spike_feats[1:4000,:]

for i in eachindex(input_words)
    params = tt.LKD.InputParams(
        dialects=[2], 
        samples=1, 
        gender=['f'], 
        words=input_words[i:i], 
        repetitions=1, # epochs 
        shift_input=2, 
        encoding="bae"
    );
    println("presenting '$(input_words[i:i])'");
    test_input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP());
    test_snn = tt.inject(test_input, snn)
    test_output = tt.test(test_snn)
end


push!(df_class, Dict(:InputParam => params, :score => score, :feature_type => "spikes", :learning_rule => "voltage"))
df_class
# run classifier on words


jldopen(folder_df, "w") do file
    file["dataframe_classifier"] = df_class
end;














##############
# TripletSTDP

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

