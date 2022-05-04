using YAML
using tt

conf = YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
path_dataset = conf["dataset_path"]

weights_params = tt.LKD.WeightParams()
params = tt.LKD.InputParams(
    dialects=[1], 
    samples=10, 
    gender=['m'], 
    words=swords[i:i], 
    repetitions=3, 
    shift_input=2, 
    encoding="bae"
)

input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.TripletSTDP());
snn = tt.SNNLayer(input);
snn_out = tt.train(snn);

input = tt.InputLayer(params, weights_params, path_dataset, path_storage, tt.VoltageSTDP());
snn = tt.SNNLayer(input);
snn_out = tt.train(snn);