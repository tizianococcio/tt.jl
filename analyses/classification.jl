using YAML
using Plots
using SpikeTimit
using LKD
using tt
using Statistics
using JLD2


conf = YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
path_dataset = conf["dataset_path"]
path_storage = conf["training_storage_path"]

# Prepare params
swords = ["the", "a", "water", "greasy"]
swords_new_stimuli = ["does", "may", "color", "easy"]
params_training = []
params_test = []
for i in 1:length(swords)
    params = tt.LKD.InputParams(
        dialects=[1], 
        samples=10, 
        gender=['m'], 
        words=swords[i:i], 
        repetitions=3, 
        shift_input=2, 
        encoding="bae"
    )
    push!(params_training, params)

    params = tt.LKD.InputParams(
        dialects=[1], 
        samples=10, 
        gender=['m'], 
        words=swords_new_stimuli[i:i], 
        repetitions=3, 
        shift_input=2, 
        encoding="bae"
    )
    push!(params_test, params)
end


## ----------------------------------------------- ## -----------------------------------------------
## Protocol 1 (a)
weights_params = tt.LKD.WeightParams()

scores = []
paths_tri = []
paths_vol = []
for i in 1:length(swords)

    params = tt.LKD.InputParams(
        dialects=[1], 
        samples=10, 
        gender=['m'], 
        words=swords[i:i], 
        repetitions=3, 
        shift_input=2, 
        encoding="bae"
    )

    # Train (learning on)
    println(">>> Training on word $(swords[i:i])")
    input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.TripletSTDP());
    snn = tt.SNNLayer(input);
    snn_out = tt.train(snn);
    
    # Test (learning off)
    println(">>> Simulation with STDP off ($(swords[i:i]))")
    snn.weights = snn_out.weights;
    snn_out = tt.test(snn);

    # Classifier (on spikes)
    println(">>> Classifier ($(swords[i:i])")
    classifier = tt.ClassificationLayer(snn_out);
    score, _ = tt.on_spikes(classifier, input)
    push!(scores, score)
    push!(paths_tri, input.store.folder)
end

# Note: execution took a very long time (35min)!
folder = joinpath(conf["experiments"], "scores_protocol_1.jld2")
jldopen(folder, "w") do file
    file["triplet/1"] = testread["triplet"]
    file["triplet/2"] = scores
    file["voltage/1"] = testread["voltage"]
end
testread = load(folder, "triplet/1")

###########################
# Now for VoltageSTDP (b)

scores = []
for i in 1:length(swords)

    params = tt.LKD.InputParams(
        dialects=[1], 
        samples=10, 
        gender=['m'], 
        words=swords[i:i], 
        repetitions=3, 
        shift_input=2, 
        encoding="bae"
    )

    println(">>> Training on word $(swords[i:i])")
    # Train (learning on)
    input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP());
    snn = tt.SNNLayer(input);
    snn_out = tt.train(snn);
    
    println(">>> Simulation with STDP off ($(swords[i:i]))")
    # Test (learning off)
    snn.weights = snn_out.weights;
    snn_out = tt.test(snn);

    println(">>> Classifier ($(swords[i:i])")
    # Classifier (on spikes)
    classifier = tt.ClassificationLayer(snn_out);
    score, _ = tt.on_spikes(classifier, input)
    push!(scores, score)
    push!(paths_vol, input.store.folder)
end

folder
jldopen(folder, "a+") do file
    file["voltage"] = scores
end
testread = load(folder)


###########################
# Now present the network with a novel stimulus (c)

# See if I can recover the saved network so I don't have to rerun the training

# load old network
input = tt.InputLayer(params_training[1], weights_params, path_dataset, path_storage, tt.TripletSTDP());
new_input = tt.InputLayer(params_test[1], weights_params, path_dataset, path_storage, tt.TripletSTDP());
old_net = tt.load(input);
@assert old_net.snn_layer.spikes_dt != new_input.spikes_dt

# inject new stimulus into old network
tt.inject(new_input, old_net.snn_layer)
@assert old_net.snn_layer.spikes_dt == new_input.spikes_dt
snn_out = tt.test(old_net.snn_layer);


# Some plotting as security checks
scatter(SpikeTimit.get_raster_data(snn_out.firing_times), m=(1, :black, stroke(0)), leg = :none, yticks=false, xtickfontsize=5, labelfontsize=5, xlabel="Time (ms)", ylabel="Neurons", grid=false, title="triplet")
plot(snn_out.firing_rates[1,:], leg=false, c=:black, tickfontsize=5, labelfontsize=5, 
xlabel="Time (ms) dt", ylabel="Hz", grid=false, title="triplet")

inputv = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP());
snnv = tt.SNNLayer(inputv)
snn_outv = tt.train(snnv)
scatter(SpikeTimit.get_raster_data(snn_outv.firing_times), m=(1, :black, stroke(0)), leg = :none, yticks=false, xtickfontsize=5, labelfontsize=5, xlabel="Time (ms)", ylabel="Neurons", grid=false, title="voltage")
classifier = tt.ClassificationLayer(snn_outv)
score, cp = tt.on_spikes(classifier, inputv)

