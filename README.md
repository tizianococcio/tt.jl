# tt.jl
## Wrapper for the LKD.jl and SpikeTimit.jl packages

### Overview
This package provides access to the main functionalities of the aforementioned packages. It provides an abstraction into a layered architecture. The theoretical layers are an input layer, a SNN layer, a classification layer.

### Minimal working example

```
using tt
params = tt.LKD.InputParams(
    dialects=[1], 
    samples=10, 
    gender=['f'], 
    words=["that", "she", "all"], 
    repetitions=9, 
    shift_input=2, 
    encoding="bae"
)
wp = tt.LKD.WeightParams()
input = tt.InputLayer(params, wp, tt.TripletSTDP())
snn = tt.SNNLayer(input)
snn_trained = tt.train(snn) # STDP-adaptation
snn_test = tt.test(snn) # simulation without STDP
classification = tt.words_classifier(input.id, snn_test)
```


### Description of abstraction layers
#### Input layer (src/input_layer.jl)
A wrapper of the input stimulus.

Can be generated by using the following method:

`InputLayer(params::LKD.InputParams, weights_params::LKD.WeightParams, stdp::STDP)`

alternatively, instead of the InputParams, the DataFrame containing the TIMIT the dataset can be used.
Additionally, the `makeinput()`and `newlike()` provide alternative interfaces to generating input layer. In particular,

`makeinput()` creates two input layers, one for a Triplet-STDP simulation, the other for a Voltage-STDP simulation, making sure the input remains the same.

The STDP argument must be either a TripletSTDP() or a VoltageSTDP(). The former allows to set the parameters values of the triplet rule, while the latter is a dummy type defined to facilitate multiple dispatch between the triplet and the voltage simulation functions.

The input layer generates a unique hash identifying the set of input parameters, this is called `id`. 

#### SNN layer (src/snn_layer.jl)
Encapsulates calling of simulations and their return values. Its main methods are the train() and test() function used to run a simulation with stdp on and off respectively. These methods return an `SNNOut` object that wraps the following properties:

- `weights`, matrix at simulation end;
- `firing_times`, vector of vectors;
- `firing_rates`, matrix of firing rates;
- `phone_states`, vector of phone states;
- `word_states`, vector of word states;
- `trackers`::Union{trackers_voltage, trackers_triplet_basic, trackers_triplet, TrackersT}
The trackers type changes according to the type of simulation run, this grew organically as the code was being written and it's far from optimal. The goal would be to standardize this using the `TrackersT` type defined in tt.jl. For now, these are the other possible return variables (defined in snn_layer.jl):

- trackers_voltage is a tuple of voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker, u, v, weight_tracker (voltage rule)
- trackers_triplet_basic is a tuple of voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker
- trackers_triplet is a tuple of voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker, r1, o1, r2, o2, weight_tracker

#### Classification layer (src/classification_layer.jl)
Provides access to the multiple linear regression classifier, classification is done using the network states (words/phones) as features.

This layer can be instantiated in two ways:

1) `ClassificationLayer(id::String, snn_out::tt.SNNOut)`
where the id is the id generated by the input layer;

2) `ClassificationLayer(id::String, exp_name::String, snn_out::tt.SNNOut)`
allowing to define an experiment name, this is useful when wanting to run and store multiple classification trials of a network with the same input.

However, for convenience a classifier on words states or phones states can be used as such:

`words_classifier(id, name::String, out::tt.SNNOut)`, where `name` is optional and represents the experiment name, as described above.
This method returns a named tuple with the following data: `accuracy, kappa, y, y_hat, labels`.

The method `phones_classifier()` behaves identically.

The weights of the classifier are automatically saved onto a file and reloaded if the same classifier on the same set of input parameters is instantiated.


### Other files

#### evaluation.jl
Wraps linear classification and generates confusion matrices.

#### utils.jl
Most of the functions have self-explanatory filenames. The functions I used the most are:

##### savexpdata(exp_name::String, what::String; data...)
Saves experimental data in JLD2 format. `exp_name` and `what` are used for the file name.

##### loadexpdata(exp_name::String, what::String)
Loads data saved with savexpdata().

#### experiment.jl
An attempt to store results from simulations using an incremental DataFrame. It turned out not to be an ideal approach since the files quickly grew in size and became slow to handle.

#### frame.jl
Contains the core functions of the package, hence "the frame". Provides path to folders, functionalities to save figures, retrieve the TIMIT dataset, access a saved SNN, as well as some functions to be used with the Experiment structure. Perhaps some functions in this file would be better off somewhere else.

#### parameters.jl
The Triplet parameters

#### plots.jl
A collection of plotting functions. Some of these are early attempts, and most of the newer figures are in the scripts folder of the Thesis repository.

#### triplet*.jl and voltage*.jl
The actual SNN simulation for the triplet and voltage rule respectively.
For the triplet rule, the simulation can be instantiated with different options, each simulation is in a different file:
- triplet_nostdp.jl: the simulation without excitatory STDP
- triplet_traces_alt.jl: simulation that allows to track pre- and post-synaptic weights for an arbitrary number of neurons (`ntrack`)
- triplet_traces.jl: simulation that tracks pre- and post-synaptic weights for one neuron
- triplet.jl: simulation that does not track weights throughout time
- voltage_m.jl: stores all simulation data in a single JLD2 file, instead of using separate files
