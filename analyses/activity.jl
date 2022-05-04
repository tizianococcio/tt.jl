## Testing stuff
tracker = zeros(100,3)

for t in 1:100
    v = rand(10)
    tracker[t,:] = v[1:3]
end

# first column
tracker[:,1]

# first row
tracker[1,:]




## Membrane potential of 3 neurons
## ----------------------------------------------- ## -----------------------------------------------
using LKD
using tt
using YAML

conf = YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
path_dataset = conf["dataset_path"]


params = tt.LKD.InputParams(
    dialects=[1,2,3,4,5,6,7,8], 
    samples=1, 
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


ids = SpikeTimit.mix_inputs(length(word_inputs.durations), params.repetitions)
durations, spikes, labels, paths = word_inputs
ordered_spikes, transcripts = SpikeTimit.get_ordered_spikes(word_inputs, ids=[1], silence_time=params.silence_time, shift=params.shift_input)
projections = LKD.ProjectionParams(npop = SpikeTimit.get_encoding_dimension(word_inputs.spikes))
weights_params = LKD.WeightParams()
W, popmembers = LKD.create_network(weights_params, projections)
weights_params.Ne
# Naming convention: LearningRule_Encoding_NumWords_d(NumDialects)s(NumSamples)r(NumRepetitions)sh(ShiftInput)g(NumGenders)_Ne_Ni
filename_inputs = "$(params.encoding)_$(length(params.words))_d$(length(params.dialects))s$(params.samples)r$(params.repetitions)sh$(round(Int, params.shift_input))g$(length(params.gender))_$(weights_params.Ne)_$(weights_params.Ni)"

# to save the post-training network
folder_name = "TRI_$(filename_inputs)"

store = LKD.StoreParams(folder = joinpath(conf["training_storage_path"], folder_name), save_states=true);
folder = store.folder
folder = LKD.makefolder(store.folder);
folder = LKD.cleanfolder(store.folder);

last_interval = ceil(transcripts.phones.intervals[end][end]*1000);
net = LKD.NetParams(simulation_time = last_interval, learning=true);
trip = tt.TripletSTDP();
transcriptions_dt = SpikeTimit.transcriptions_dt(transcripts);
spikes_dt = SpikeTimit.ft_dt(ordered_spikes);

W, T, R, trackers = tt.sim(W, popmembers, spikes_dt, transcriptions_dt, net, store, weights_params, projections, trip);

voltage, adapt_curr, adapt_threshold = trackers
sim_time_dt = round(Int, last_interval*10)
plot(voltage[:,1])
for id in eachindex(spikes_dt.neurons)
    fn = intersect(spikes_dt.neurons[id], [1])
    if length(fn) > 0
        l = vline!([spikes_dt.ft[id]], c=:black)
        display(l)
    end
end
plot!(leg=false)

plot(adapt_curr[:,1], grid=false, leg=false)
plot(adapt_threshold[:,1], grid=false, leg=false)