@with_kw mutable struct ClassificationLayer
    firing_times::Vector{Vector{Float64}}
    firing_rates::Matrix{Float32}
    phone_states::Vector{Any}
    word_states::Vector{Any}
end

function ClassificationLayer(snn_out::tt.SNNOut)
    ClassificationLayer(
        snn_out.firing_times,
        snn_out.firing_rates,
        snn_out.phone_states,
        snn_out.word_states
    )
end

function _states_classifier(states::Vector{Any})
    feats, n_neurons, labels = LKD.states_to_features(states)
    score, params = LKD.MultiLogReg(feats, labels)
    score, params
end

function on_phones(classLayer::ClassificationLayer)
    _states_classifier(classLayer.phone_states)
end

function on_words(classLayer::ClassificationLayer)
    _states_classifier(classLayer.word_states)
end

function on_spikes(classLayer::ClassificationLayer, input::InputLayer)
    interval = 100:100.:input.net.simulation_time
    spike_feats, spike_labels = LKD.spikes_to_features(classLayer.firing_times, input.transcriptions.phones, interval)
    LKD.MultiLogReg(spike_feats[1:input.weights_params.Ne,:], spike_labels)
end
