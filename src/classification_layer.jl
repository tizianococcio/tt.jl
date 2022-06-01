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

function predict(X, y, theta)
    y_numeric = tt.LKD.labels_to_y(y)
    n_classes = length(Set(y))
    train_std = StatsBase.fit(StatsBase.ZScoreTransform, X, dims = 2)
    StatsBase.transform!(train_std, X)
    preds = MLJLinearModels.softmax(MLJLinearModels.apply_X(X', theta, n_classes))
    y_hat = map(x -> argmax(x), eachrow(preds))
    return (y=y_numeric, y_hat=y_hat, n_classes=n_classes)
end

function compute_confmat(y, y_hat)
    n_classes = length(Set(y))
    cm = zeros(n_classes, n_classes)
    for (truth, prediction) in zip(y, y_hat)
        cm[truth, prediction] += 1
    end
    return cm
end

function accuracy(y, y_hat)
    @assert length(y) == length(y_hat)
    mean(y .== y_hat)
end