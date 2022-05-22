using Revise
import SpikeTimit
using Random
import LKD
using YAML

conf = YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
path_dataset = conf["dataset_path"]

train_path = joinpath(path_dataset, "train");
dict_path = joinpath(path_dataset, "DOC", "TIMITDIC.TXT");
train_df = SpikeTimit.create_dataset(;dir= train_path)
dict = SpikeTimit.create_dictionary(file=dict_path)

words = ["that", "she", "all", "your", "had" ]
inputs = LKD.InputParams(dialects=[1], samples=20, gender=['m'], words=words
, repetitions=3, shift_input=2, encoding="cochlea70")
filtered_df = filter(:words => x-> any([word in x for word in inputs.words]), train_df) |> df -> filter(:dialect => x->x ∈ inputs.dialects, df) |> df -> filter(:gender => x->x ∈ inputs.gender, df)
word_inputs = SpikeTimit.select_words(filtered_df, samples=inputs.samples, inputs.words, encoding= inputs.encoding )
if inputs.encoding == "bae"
	SpikeTimit.resample_spikes!(word_inputs.spikes)
	SpikeTimit.transform_into_bursts!(word_inputs.spikes)
end
ids = SpikeTimit.mix_inputs(length(word_inputs.durations), 3)
spikes, transcriptions=  SpikeTimit.get_ordered_spikes(word_inputs, ids=ids, silence_time=inputs.silence_time, shift = inputs.shift_input)
transcriptions_dt = SpikeTimit.transcriptions_dt(transcriptions)
spikes_dt = SpikeTimit.ft_dt(spikes)
last_interval = ceil(transcriptions.phones.intervals[end][end]*1000)
net = LKD.NetParams(simulation_time = last_interval, learning=true)
weights_params = LKD.WeightParams()
projections = LKD.ProjectionParams(
		npop = SpikeTimit.get_encoding_dimension(word_inputs.spikes))
store = LKD.StoreParams(folder = joinpath(conf["training_storage_path"], "cochlea_5_words"), save_timestep=10_000)
folder = store.folder
W, popmembers = LKD.create_network(weights_params, projections)
folder = LKD.makefolder(store.folder);
folder = LKD.cleanfolder(store.folder);
LKD.save_network(popmembers, W, folder)
Random.seed!(inputs.random_seed)

################################################ STAGE 1 ##########################################
# STDP on
res = LKD.sim(W, popmembers, spikes_dt, transcriptions_dt, net, store, weights_params, projections)
res

################################################ STAGE 2 ##########################################
# STDP off
folder = joinpath(conf["training_storage_path"], "cochlea_5_words")
Wr = LKD.read_network_weights(folder);
W
Wl = Wr[1][2]
W = copy(Wl)
net.learning = false
store.folder = joinpath(store.folder, "stdp_off")
folder = LKD.makefolder(store.folder);
folder = LKD.cleanfolder(store.folder);
res_off = LKD.sim(W, popmembers, spikes_dt, transcriptions_dt, net, store, weights_params, projections)

net_spikes = LKD.read_network_spikes(folder);
net_rates = LKD.read_network_rates(folder);
word_states = LKD.read_network_states(joinpath(folder,"word_states"));
phone_states = LKD.read_network_states(joinpath(folder,"phone_states"));

# Classification
using StatsBase, MLJLinearModels, JLD2

# Preprocessing
X, n_neurons, labels = LKD.states_to_features(word_states)
y = LKD.labels_to_y(labels)
n_classes = length(Set(labels))
n_features = size(X, 1)
n_samples = size(X, 2)
train, test = LKD.train_test_indices(labels, 0.7)
train_std = StatsBase.fit(ZScoreTransform, X[:, train], dims = 2)
StatsBase.transform!(train_std, X)

# MultinomialRegression
λ = 0.5
mnr = MultinomialRegression(Float64(λ); fit_intercept = false)
θ = MLJLinearModels.fit(mnr, X[:, train]', y[train])
preds = MLJLinearModels.softmax(MLJLinearModels.apply_X(X[:, test]', θ, n_classes))
targets = map(x -> argmax(x), eachrow(preds))
scores = mean(targets .== y[test])
params = reshape(θ, n_features, n_classes) # params: 8000x5 matrix
classifier_path = joinpath(folder, "trained_classifier.jld2")
jldopen(classifier_path, "w") do file
    file["params"] = params
end

# load trained model
folder = joinpath(conf["training_storage_path"], "cochlea_5_words")
params = JLD2.load(joinpath(folder, "trained_classifier.jld2"), "params")


# now run the simulation with a new input ("your")
inputs = LKD.InputParams(dialects=[1], samples=20, gender=['m'], words=["your"], repetitions=1, shift_input=2, encoding="cochlea70")
train_df = SpikeTimit.create_dataset(;dir= train_path)
filtered_df = filter(:words => x-> any([word in x for word in inputs.words]), train_df) |> df -> filter(:dialect => x->x ∈ inputs.dialects, df) |> df -> filter(:gender => x->x ∈ inputs.gender, df)
word_inputs = SpikeTimit.select_words(filtered_df, samples=inputs.samples, inputs.words, encoding= inputs.encoding )
if inputs.encoding == "bae"
	SpikeTimit.resample_spikes!(word_inputs.spikes)
	SpikeTimit.transform_into_bursts!(word_inputs.spikes)
end
ids = SpikeTimit.mix_inputs(length(word_inputs.durations), 3)
spikes, transcriptions=  SpikeTimit.get_ordered_spikes(word_inputs, ids=ids, silence_time=inputs.silence_time, shift = inputs.shift_input)
transcriptions_dt = SpikeTimit.transcriptions_dt(transcriptions)
spikes_dt = SpikeTimit.ft_dt(spikes)
last_interval = ceil(transcriptions.phones.intervals[end][end]*1000)
net = LKD.NetParams(simulation_time = last_interval, learning=false)
weights_params = LKD.WeightParams()
store.folder = joinpath(conf["training_storage_path"], "cochlea_5_words", "new_input_your")
folder = LKD.makefolder(store.folder);
folder = LKD.cleanfolder(store.folder);
LKD.sim(W, popmembers, spikes_dt, transcriptions_dt, net, store, weights_params, projections)
a

# collect the features (word states)
folder
word_states = LKD.read_network_states(joinpath(folder,"word_states"));

# feed the states (transformed into features) to the trained classifier: apply softmax using the theta parameters
X, n_neurons, lab2 = LKD.states_to_features(word_states)
preds = MLJLinearModels.softmax(MLJLinearModels.apply_X(X', θ, n_classes))


# retrieve the prediction with highest probability, the presented word should have the highest probability
targets = map(x -> argmax(x), eachrow(preds))
sum(targets .== 4)



# now run the simulation with a new input ("all")
inputs = LKD.InputParams(dialects=[1], samples=20, gender=['m'], words=["all"], repetitions=1, shift_input=2, encoding="cochlea70")
train_df = SpikeTimit.create_dataset(;dir= train_path)
filtered_df = filter(:words => x-> any([word in x for word in inputs.words]), train_df) |> df -> filter(:dialect => x->x ∈ inputs.dialects, df) |> df -> filter(:gender => x->x ∈ inputs.gender, df)
word_inputs = SpikeTimit.select_words(filtered_df, samples=inputs.samples, inputs.words, encoding= inputs.encoding )
if inputs.encoding == "bae"
	SpikeTimit.resample_spikes!(word_inputs.spikes)
	SpikeTimit.transform_into_bursts!(word_inputs.spikes)
end
ids = SpikeTimit.mix_inputs(length(word_inputs.durations), 3)
spikes, transcriptions=  SpikeTimit.get_ordered_spikes(word_inputs, ids=ids, silence_time=inputs.silence_time, shift = inputs.shift_input)
transcriptions_dt = SpikeTimit.transcriptions_dt(transcriptions)
spikes_dt = SpikeTimit.ft_dt(spikes)
last_interval = ceil(transcriptions.phones.intervals[end][end]*1000)
net = LKD.NetParams(simulation_time = last_interval, learning=false)
weights_params = LKD.WeightParams()
store.folder = joinpath(conf["training_storage_path"], "cochlea_5_words", "new_input_all")
folder = LKD.makefolder(store.folder);
folder = LKD.cleanfolder(store.folder);
LKD.sim(W, popmembers, spikes_dt, transcriptions_dt, net, store, weights_params, projections)
a

# collect the features (word states)
folder = joinpath(conf["training_storage_path"], "cochlea_5_words", "new_input_all")
word_states = LKD.read_network_states(joinpath(folder,"word_states"));
word_states
# feed the states (transformed into features) to the trained classifier: apply softmax using the theta parameters
X1, n_neurons, lab2 = LKD.states_to_features(word_states)
n_classes = 5
size(params, 1)
θ = reshape(params, size(params, 1)*n_classes)
train_std = StatsBase.fit(ZScoreTransform, X1, dims = 2)
StatsBase.transform!(train_std, X1)
preds = MLJLinearModels.softmax(MLJLinearModels.apply_X(X1', θ, n_classes))
words
θ
n_neurons
lab2

# retrieve the prediction with highest probability, the presented word should have the highest probability
targets = map(x -> argmax(x), eachrow(preds))
sum(targets .== 3)

