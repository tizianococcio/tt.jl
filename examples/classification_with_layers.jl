using tt

conf = tt.load_conf()
path_dataset = conf["dataset_path"]
path_storage = conf["training_storage_path"]
path_experiments = conf["experiments"]

input_words = ["that", "she", "all", "your", "had" ]
params = tt.LKD.InputParams(
    dialects=[1], 
    samples=20, 
    gender=['m'], 
    words=input_words, 
    repetitions=3, 
    shift_input=2, 
    encoding="cochlea70"
);
weights_params = tt.LKD.WeightParams();
input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP());
snn = tt.SNNLayer(input);
snn.store.folder

_, out_stdpoff, _ = tt.load(input)
out_stdpoff
################################################ STAGE 1 ##########################################
# STDP on
out = tt.train(snn)
out.word_states
################################################ STAGE 2 ##########################################
# STDP off
out_stdpoff = tt.test(snn)
out.word_states == out_stdpoff.word_states


# Classification
using StatsBase, MLJLinearModels, JLD2

# Preprocessing
classifier_path = joinpath(snn.store.folder, "trained_classifier.jld2")
θ = JLD2.load(joinpath(classifier_path), "theta")
X, n_neurons, labels = tt.LKD.states_to_features(out_stdpoff.word_states)
y = tt.LKD.labels_to_y(labels)
n_classes = length(Set(labels))
n_features = size(X, 1)
n_samples = size(X, 2)
ttrain, test = LKD.train_test_indices(labels, 0.7)
train_std = StatsBase.fit(ZScoreTransform, X, dims = 2)
StatsBase.transform!(train_std, X)

# MultinomialRegression (classifier training)
λ = 0.5
mnr = MultinomialRegression(Float64(λ); fit_intercept = false)
θ = MLJLinearModels.fit(mnr, X', y)
jldopen(classifier_path, "w") do file
    file["theta"] = θ
end

# this first prediction run should give 100%
preds = MLJLinearModels.softmax(MLJLinearModels.apply_X(X', θ, n_classes))
targets = map(x -> argmax(x), eachrow(preds))
scores = mean(targets .== y)

# now run the simulation with a new input ("your")
params = tt.LKD.InputParams(
    dialects=[1], 
    samples=5, 
    gender=['m'], 
    words=["your"], 
    repetitions=3, # epochs 
    shift_input=4, 
    encoding="cochlea70"
);
weights_params = tt.LKD.WeightParams();
input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP());
snn_your = tt.SNNLayer(input);
snn_your.store.folder
sno = tt.load(snn_your.store.folder)

snn_your.weights = copy(out_stdpoff.weights)
snn_your.popmembers = copy(snn.popmembers)
out_your = tt.test(snn_your)

# check that word states are different from previous run
@assert out_your.word_states != out_stdpoff.word_states

# classify states, expecting 100% for "your"
X_your, _, _ = tt.LKD.states_to_features(sno.word_states)
train_std = StatsBase.fit(ZScoreTransform, X_your, dims = 2)
StatsBase.transform!(train_std, X_your)
preds = MLJLinearModels.softmax(MLJLinearModels.apply_X(X_your', θ, n_classes))
targets = map(x -> argmax(x), eachrow(preds))
scores_your = [mean(targets .== i) for i in 1:n_classes]
mean(targets .== 3)


# now run the simulation with a new input ("had")
params = tt.LKD.InputParams(
    dialects=[1], 
    samples=5, 
    gender=['m'], 
    words=["had"], 
    repetitions=3, # epochs 
    shift_input=2, 
    encoding="cochlea70"
);
weights_params = tt.LKD.WeightParams();
input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP());
snn_had = tt.SNNLayer(input);
snn_had.store.folder
snn_had.weights = copy(out_stdpoff.weights)
snn_had.popmembers = copy(snn.popmembers)
out_had = tt.test(snn_had)

# check that word states are different from previous run
@assert out_had.word_states != out_stdpoff.word_states

# classify states, expecting 100% for "had"
X_had, _, _ = tt.LKD.states_to_features(out_had.word_states)
train_std = StatsBase.fit(ZScoreTransform, X_had, dims = 2)
StatsBase.transform!(train_std, X_had)
preds = MLJLinearModels.softmax(MLJLinearModels.apply_X(X_had', θ, n_classes))
targets = map(x -> argmax(x), eachrow(preds))
scores_had = [mean(targets .== i) for i in 1:n_classes]

# loop thru all words, classify them in turn
scores = []
for wordid in eachindex(input_words)
    println("Classifying $(input_words[wordid:wordid])")
    params = tt.LKD.InputParams(
        dialects=[1], 
        samples=20, 
        gender=['m'], 
        words=input_words[wordid:wordid], 
        repetitions=3, # epochs 
        shift_input=2, 
        encoding="cochlea70"
    );
    weights_params = tt.LKD.WeightParams();
    input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP());
    snn_your = tt.SNNLayer(input);
    snn_your.weights = copy(out_stdpoff.weights)
    snn_your.popmembers = copy(snn.popmembers)
    out_your = tt.test(snn_your)

    # check that word states are different from previous run
    @assert out_your.word_states != out_stdpoff.word_states

    # classify states, expecting 100% for "your"
    X_your, _, _ = LKD.states_to_features(out_your.word_states)
    train_std = StatsBase.fit(ZScoreTransform, X_your, dims = 2)
    StatsBase.transform!(train_std, X_your)
    preds = MLJLinearModels.softmax(MLJLinearModels.apply_X(X_your', θ, n_classes))
    targets = map(x -> argmax(x), eachrow(preds))
    scores_your = [mean(targets .== i) for i in 1:n_classes]
    push!(scores, scores_your)
end


function predict(data, params, n_classes)
    X, n_neurons, lab = LKD.states_to_features(data)
    MLJLinearModels.softmax(MLJLinearModels.apply_X(X', reshape(params, size(params, 1)*n_classes), n_classes))
end
