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
weights_params = tt.LKD.WeightParams()
swords = ["the", "a", "water", "greasy"]
swords_new_stimuli = ["does", "may", "speaker", "easy"]
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
        dialects=[1,2,3,4,5,6,7], 
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
folder = joinpath(conf["experiments"], "scores_protocol_1.jld2")

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
jldopen(folder, "w") do file
    file["triplet/2"] = scores
end
testread = load(folder, "triplet/1")

"""
jld2 file structure

+ root
  voltage
    1 (first trial)
    ...
  triplet
    2 (second trial)
    ...
"""


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

jldopen(folder, "a+") do file
    file["voltage"] = scores
end
testread = load(folder)



###########################
# Now present the network with a novel stimulus (c)
scores_triplet = []
# loop all words
for i in 1:length(swords)
    println("Word $(swords[i])");
    # load old network
    input = tt.InputLayer(params_training[i], weights_params, path_dataset, path_storage, tt.TripletSTDP());
    old_net = tt.load(input);
    # load new input
    new_input = tt.InputLayer(params_test[i], weights_params, path_dataset, path_storage, tt.TripletSTDP());
    # check that they are indeed different
    @assert old_net.snn_layer.spikes_dt != new_input.spikes_dt

    # inject new stimulus into old network
    tt.inject(new_input, old_net.snn_layer)
    # check that old network has now the new stimulus
    @assert old_net.snn_layer.spikes_dt == new_input.spikes_dt
    # run simulation with training off
    snn_out = tt.test(old_net.snn_layer);

    # run classifier
    classifier = tt.ClassificationLayer(snn_out);
    score, _ = tt.on_spikes(classifier, new_input);
    push!(scores_triplet, score);

end

# repeat for voltage
scores_voltage = []
for i in 1:length(swords)
    # load old network
    input = tt.InputLayer(params_training[i], weights_params, path_dataset, path_storage, tt.VoltageSTDP());
    old_net = tt.load(input);
    # load new input
    new_input = tt.InputLayer(params_test[i], weights_params, path_dataset, path_storage, tt.VoltageSTDP());
    # check that they are indeed different
    @assert old_net.snn_layer.spikes_dt != new_input.spikes_dt

    # inject new stimulus into old network
    tt.inject(new_input, old_net.snn_layer)
    # check that old network has now the new stimulus
    @assert old_net.snn_layer.spikes_dt == new_input.spikes_dt
    # run simulation with training off
    snn_out = tt.test(old_net.snn_layer);

    # run classifier
    classifier = tt.ClassificationLayer(snn_out);
    score, _ = tt.on_spikes(classifier, new_input);
    push!(scores_voltage, score);

end

jldopen(folder, "a+") do file
    file["novel/triplet"] = scores_triplet
    #file["novel/voltage"] = scores_voltage
end

# Plot the 4 words trained and tested on the same network both for triplet and voltage.
# Then plot the new input (the other 4 words) tested on networks trained above, both for voltage and triplet.
using StatsPlots
exp_results = hcat(data["voltage/1"], data["triplet/1"])


data = load(folder)
labels = repeat(["VoltageSTDP", "TripletSTDP"], inner=length(swords))
xlabels = repeat(swords, outer=2)

groupedbar(xlabels, exp_results, 
    group=labels, bar_width=0.7, lw=1, framestyle=:box, 
    xlabel="Input word", ylabel="Classification accuracy", 
    c = :black, grid=false, legend=:bottomleft; fillstyle = [nothing :x])
a = 1


# Some plotting as security checks
scatter(SpikeTimit.get_raster_data(snn_out.firing_times), m=(1, :black, stroke(0)), leg = :none, yticks=false, xtickfontsize=5, labelfontsize=5, xlabel="Time (ms)", ylabel="Neurons", grid=false, title="triplet")
plot(snn_out.firing_rates[1,:], leg=false, c=:black, tickfontsize=5, labelfontsize=5, 
xlabel="Time (ms) dt", ylabel="Hz", grid=false, title="triplet")
