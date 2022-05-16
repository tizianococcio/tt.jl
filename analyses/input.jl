using YAML
using Plots
using SpikeTimit
using LKD
using tt
using Statistics
using JLD2

conf = YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
path_dataset = conf["dataset_path"]

train = tt.get_timit_train_dataframe(path_dataset)
dict = SpikeTimit.create_dictionary(file=joinpath(path_dataset, "DOC", "TIMITDIC.TXT"));

# just to find the most common word: maybe will be useful later
counts = Dict{String, Int32}()
for s in train.sentence
    for word in s
        if !haskey(counts, word)
            counts[word] = 0
        end
        counts[word] += 1
    end
end
counts = sort(collect(counts), by = tuple -> last(tuple), rev=true);
counts[60:70]
counts["may"]

params = tt.LKD.InputParams(
    dialects=[1,2,3,4,5,6,7,8], 
    samples=8, 
    gender=['f','m'], 
    words=["music", "rag"], 
    repetitions=3, 
    shift_input=2, 
    encoding="cochlea70"
)
df = tt.load_dataset(path_dataset, params)
word_inputs = SpikeTimit.select_words(
    df, 
    samples=params.samples, 
    params.words, 
    encoding=params.encoding)

duration = word_inputs[1] # duration (1 array per word)
spikes = word_inputs[2] # spikes (1 array per word)
labels = word_inputs[3] # labels (1 array per word)



## Data for Experiment E
## ----------------------------------------------- ## -----------------------------------------------
## plot 4 input words (spike representation) both encoded with cochlea and with bae and the respective network representation after 10 passess through the data for both the triplet and the voltage stdp network

# words counts in training data set
# "the" => 1603
# "to" => 1018
# "in" => 947
#  "a" => 867
# "that" => 612
# "she" => 572


function exp_E(encoding::String)
    weights_params = tt.LKD.WeightParams()
    conf = YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
    path_dataset = conf["dataset_path"]
    path_storage = conf["training_storage_path"]
    swords = ["the", "a", "water", "greasy"]
    swords[1:1]
    
    # to store plotting data
    in_spikes = []
    sim_triplet = []
    sim_voltage = []
    for i in 1:length(swords)
        params = tt.LKD.InputParams(
            dialects=[1], 
            samples=1, 
            gender=['m'], 
            words=swords[i:i], 
            repetitions=1, 
            shift_input=100, # ms
            encoding=encoding
        )
        df = tt.load_dataset(path_dataset, params)
        word_inputs = SpikeTimit.select_words(
            df, 
            samples=params.samples, 
            params.words, 
            encoding=params.encoding);
    
        # getting the spikes here
        duration, spikes, labels = word_inputs
        push!(in_spikes, spikes)
    
        # now run simulation 10 times for each input and type (triplet/voltage), get raster and firing rates
        # TripletSTDP
        input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.TripletSTDP())
        snn = tt.SNNLayer(input)
        
        raster_triplet = []
        frs_triplet = []
        println("triplet simulations for word '$(swords[i])'")
        for j in 1:10
            println("iteration $j")
            out = tt.train(snn);
            snn.weights = out.weights;
            push!(raster_triplet, out.firing_times);
            push!(frs_triplet, out.firing_rates[1,:]);
        end
    
        # VoltageSTDP
        input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP())
        snn = tt.SNNLayer(input)
        
        raster_voltage = []
        frs_voltage = []
        println("voltage simulations for word '$(swords[i])'")
        for j in 1:10
            println("iteration $j")
            out = tt.train(snn);
            snn.weights = out.weights;
            push!(raster_voltage, out.firing_times);
            push!(frs_voltage, out.firing_rates[1,:]);
        end
    
        # Note: I save all spikes for each simulation, although I will probably only want the last one
        push!(sim_triplet, (net_spikes=raster_triplet, net_fr=frs_triplet));
        push!(sim_voltage, (net_spikes=raster_voltage, net_fr=frs_voltage));
    end
    
    # Save experiment data
    folder = joinpath(conf["experiments"], "data_exp_E_$encoding.jld2")
    fid = Dict()
    fid["sim_triplet"] = sim_triplet
    fid["sim_voltage"] = sim_voltage
    fid["in_spikes"] = in_spikes
    save(folder, fid)
end


exp_E("bae");
data = load(joinpath(conf["experiments"], "data_exp_E_bae.jld2"))

exp_E("cochlea70");
data = load(joinpath(conf["experiments"], "data_exp_E_bae.jld2"))


## PART A
## ----------------------------------------------- ## -----------------------------------------------
# Plot A1: 70-neuron cochlear encoding for one sample of the word "music"
x, y = SpikeTimit.get_raster_data(spikes[3])
part_a = scatter(x,y, m=(3, :black, stroke(0)), leg = :none, xlabel="Time (s)", ylabel="Neurons");
Plots.savefig("PartA.pdf")

# Overlapping plots using gradients, currently not used.
grad = cgrad(:lightrainbow, 11, categorical = true)
x, y = SpikeTimit.get_raster_data(spikes[2])
scatter(x,y, m=(3, :black, stroke(0)), leg=:none)
for i in 2:length(spikes)
    x, y = SpikeTimit.get_raster_data(spikes[i])
    sc = scatter!(x,y, m=(3, grad[i], stroke(0)));
    display(sc)
    print(i)
end

# other ways of getting spike encodings:
mws = SpikeTimit.get_word_spikes(df, "music"; encoding="cochlea70")
s = SpikeTimit.get_spiketimes(df=df, encoding="cochlea70")

## PART C
## ----------------------------------------------- ## -----------------------------------------------
## Plot multiple samples onto the same raster plot
## 70-neuron cochlear encoding for 8 samples of the word "music"
# points are shown with high visual transparency to show overlap between samples
x, y = SpikeTimit.get_raster_data(spikes[1])
scatter(x,y, m=(3, :black, stroke(0)), leg = :none, alpha=0.2)
for i in 2:length(spikes)
    x, y = SpikeTimit.get_raster_data(spikes[i])
    sc = scatter!(x,y, m=(3, :black, stroke(0)), leg = :none, alpha=0.2)
    display(sc)
end
part_c = scatter!(xlabel="Time (s)", ylabel="Neurons");
Plots.savefig("PartC.pdf")

## --
## Plot of spike encoding with bars delimiting the phones
duration, spikes, labels = word_inputs
sample_id = 1
#ids = SpikeTimit.mix_inputs(length(word_inputs.durations), params.repetitions)
ordered_spikes, transcripts = SpikeTimit.get_ordered_spikes(word_inputs, ids=[sample_id], silence_time=params.silence_time, shift=params.shift_input)

grad = cgrad(:roma, length(transcripts.phones.intervals), categorical = true);
x, y = SpikeTimit.get_raster_data(spikes[sample_id])
scatter(x,y, m=(3, :black, stroke(0)), leg = :none, xlabel="Time (s)", ylabel="Neurons")

i = 1
label_y = 70
label_size = 8
for interval in transcripts.phones.intervals
    from, to = interval
    label_x = ((to-from)/2)+from
    label = transcripts.phones.signs[i]
    l = vspan!([from, to], c=grad[i], alpha = 0.3)
    annotate!([(label_x, label_y, (label, label_size, :black))])
    fig = display(l)
    i += 1
end
part_c1 = plot!();
part_c2 = plot!();
savefig("PartC2.pdf")

## Put all together in a figure
lt = @layout [a b; c d]
plot(part_a, part_c, part_c1, part_c2, layout=lt, 
    xlabel=["" "" "Time (s)" "Time (s)"], ylabel=["Neurons" "" "Neurons" ""], 
    title=["A" "B" "C" "D"], titlefontsize=7, labelfontsize=6, top_margin = 0Plots.mm, left_margin = 0Plots.mm,
    titlelocation=:left, tickfontsize=7)
savefig("input_analysis.pdf")

## Here I tyr to make a plot that represents the phones overlapping with the audio signal
# But there is a problem with the fact that the audio signal contains way more samples than the spikes
params = tt.LKD.InputParams(
    dialects=[1,2,3,4,5,6,7,8], 
    samples=3, 
    gender=['f','m'], 
    words=["music"], 
    repetitions=1, 
    shift_input=2, 
    encoding="cochlea70"
)
df = tt.load_dataset(path_dataset, params)
df.path
word_inputs = SpikeTimit.select_words(
    df, 
    samples=params.samples, 
    params.words, 
    encoding=params.encoding)

durations, spikes, labels, paths = word_inputs
word_inputs.paths[ids]
ids = SpikeTimit.mix_inputs(length(word_inputs.durations), params.repetitions)
ordered_spikes, transcripts = SpikeTimit.get_ordered_spikes(word_inputs, ids=[1], silence_time=params.silence_time, shift=params.shift_input)
ordered_spikes.ft # when they fire
ordered_spikes.neurons # who fires

transcripts.words
transcripts.phones.intervals
transcripts.phones.signs

using WAV
transcripts.words
word_inputs.labels[1]
labels[1].word
local_path = "/Users/tiziano/M/Education/Ru-BScAI/Courses/3/Thesis/audio/si714.wav"
sr = 16000
t0 = labels[1].t0*sr+30000
t1 = labels[1].t1*sr+30000
snd, sampFreq = wavread(local_path, subrange=t0:t1)
wavplay(snd, sampFreq)
s1 = snd[:,1] 
d = size(s1,1)
timeArray = (0:(d-1)) / sr
timeArray = timeArray / 1000 #scale to milliseconds

x, y = SpikeTimit.get_raster_data(spikes[1])
x *= 1000
scatter(x,y, m=(3, :black, stroke(0)), leg = :none, title="A.1", xlabel="time (ms)", ylabel="neurons")
Plots.plot(twinx(), s1./1000)
#Plots.plot(timeArray, s1)



## PART B
## ----------------------------------------------- ## -----------------------------------------------
# Describe how the input layer is injected into the actual SNN. => The projection!
# At each time step of the simulation, the firing neurons of the input layer that are firing at that moment are aggregated into a population (an assembly) that I call  "firing neurons".
# When the weight matrix is generated, the neurons from the input layer are mapped onto excitatory neurons with uniform probability p=0.35. I refer to this as "population members".

# Then, each member p of this firing population will be associated to a subset of the network's excitatory neurons. Each neuron within this subset receives an input current to represent a stimulus.

params = tt.LKD.InputParams(
    dialects=[1,2,3,4,5,6,7,8], 
    samples=3, 
    gender=['f','m'], 
    words=["music"], 
    repetitions=1, 
    shift_input=2, 
    encoding="cochlea70"
)
df = tt.load_dataset(path_dataset, params)
word_inputs = SpikeTimit.select_words(
    df, 
    samples=params.samples, 
    params.words, 
    encoding=params.encoding)

durations, spikes, labels, paths = word_inputs
sample_id = 3
ordered_spikes, transcripts = SpikeTimit.get_ordered_spikes(word_inputs, ids=[sample_id], silence_time=params.silence_time, shift=params.shift_input)
projections = LKD.ProjectionParams(npop = SpikeTimit.get_encoding_dimension(word_inputs.spikes))
weights_params = LKD.WeightParams()

# How population members are computed
Ne = 4000
pmembership = projections.pmembership
Nmaxmembers = round(Int,(projections.pmembership*Ne)*1.5)
popmembers = fill(-1, Nmaxmembers, projections.npop);
size(popmembers)

W, popmembers = LKD.create_network(weights_params, projections)
size(W)
size(popmembers)

heatmap(c=cgrad([:white, :black]), W[1:100,1:100], ratio=1,showaxis=false, grid=false, ticks=false, legend=:none)
savefig("neurons.pdf")
input_layer = rand(70) .< 0.5
heatmap(c=cgrad([:black,:white]), input_layer',  xflip=true)

# spikes
ordered_spikes.ft # when they fire
ordered_spikes.neurons # who fires
durations

# words and phones
transcripts.words
transcripts.phones.intervals
transcripts.phones.signs


## visualizing projection from input layer onto SNN weight matrix -> did not manage so far
## ----------------------------------------------- ## -----------------------------------------------
params = tt.LKD.InputParams(
    dialects=[1,2,3,4,5,6,7,8], 
    samples=3, 
    gender=['f','m'], 
    words=["music"], 
    repetitions=1, 
    shift_input=2, 
    encoding="cochlea70"
)
df = tt.load_dataset(path_dataset, params)
word_inputs = SpikeTimit.select_words(
    df, 
    samples=params.samples, 
    params.words, 
    encoding=params.encoding);

durations, spikes, labels, paths = word_inputs
sample_id = 3
ordered_spikes, transcripts = SpikeTimit.get_ordered_spikes(word_inputs, ids=[sample_id], silence_time=params.silence_time, shift=params.shift_input)
projections = LKD.ProjectionParams(npop = SpikeTimit.get_encoding_dimension(word_inputs.spikes))
weights_params = LKD.WeightParams()
W, popmembers = LKD.create_network(weights_params, projections)

for assembly in popmembers[1:3]
    println(assembly)
end

popmembers[1200]
popmembers[3,4]
W .= 0
size(popmembers,1)
popmembers[2,:]

for j in 1:70
    for i in 1:size(popmembers,1)
        if j != -1
            W[j,i] = 99
        end
    end
end

heatmap(W[1:Ne,1:Ne])


## New input layer setup
## ----------------------------------------------- ## -----------------------------------------------
using YAML
using Plots
import tt

params = tt.LKD.InputParams(
    dialects=[1,2,3,4,5,6,7,8], 
    samples=1, 
    gender=['f','m'], 
    words=["music"], 
    repetitions=1, 
    shift_input=2, 
    encoding="bae"
)
weights_params = tt.LKD.WeightParams()
conf = YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
path_dataset = conf["dataset_path"]
path_storage = conf["training_storage_path"]

# TripletSTDP
input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.TripletSTDP())
snn = tt.SNNLayer(input)
W, T, R, trackers = tt.train(snn)
heatmap(W[1:weights_params.Ne,1:weights_params.Ne])

raster = [T]
frs = [R[1,:]]
# rerun
for i in 2:10
    println("iteration $i")
    snn.weights = W
    W, T, R, trackers = tt.train(snn)
    x, y = tt.SpikeTimit.get_raster_data(T)
    push!(raster, T);
    push!(frs, R[1,:]);
end

plot(frs, xlabel="Simulation time step", ylabel="Firing rate (Hz)")
#rates = plot(R[1,:], label="$i", xlabel="Simulation time step", ylabel="Firing rate (Hz)");
raster_plots = []
for i in 1:10
    spikes_raster = scatter(x,y, m=(1, :black, stroke(0)), leg = :none, xlabel="Time (ms)", ylabel="Neurons", title="$i");
    push!(raster_plots, spikes_raster);
end

plot(raster_plots..., layout=(1,10))



# VoltageSTDP
inputv = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP())
snnv = tt.SNNLayer(inputv)
Wv, Tv, Rv, trackersv = tt.train(snnv)
heatmap(Wv[1:weights_params.Ne,1:weights_params.Ne])
