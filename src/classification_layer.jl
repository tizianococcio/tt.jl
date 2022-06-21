using Random
@with_kw mutable struct ClassificationLayer
    id::String
    firing_times::Vector{Vector{Float64}}
    firing_rates::Matrix{Float32}
    phone_states::Vector{Any}
    word_states::Vector{Any}
    save_params::Bool=true
end

function ClassificationLayer(id::String, snn_out::tt.SNNOut)
    ClassificationLayer(
        id,
        snn_out.firing_times,
        snn_out.firing_rates,
        snn_out.phone_states,
        snn_out.word_states,
        true
    )
end

function clf_exists(id::String, name::String)
    isfile(joinpath(tt.rawdatadir(), "c_$(id)_$(name)"))
end
function load_clf(id::String, name::String)
    sc = nothing
    pa = nothing 
    p = joinpath(tt.rawdatadir(), "c_$(id)_$(name)")
    if isfile(p)
        file = jldopen(p, "r")
        sc = file["score"]
        pa = file["params"]
        close(file)
    end
    return sc, pa
end

function save(id::String, name::String, score, params)
    p = joinpath(tt.rawdatadir(), "c_$(id)_$(name)")
    file = jldopen(p, "w")
    file["score"] = score
    file["params"] = params
    close(file)
end

function _states_classifier(states::Vector{Any})
    feats, n_neurons, labels = LKD.states_to_features(states)
    score, params = LKD.MultiLogReg(feats, labels)
    score, params
end

function on_phones(classLayer::ClassificationLayer)
    if clf_exists(classLayer.id, "phones")
        s, p = load_clf(classLayer.id, "phones")
    else
        s, p = _states_classifier(classLayer.phone_states)
        if classLayer.save_params
            save(classLayer.id, "phones", s, p)
        end
    end
    s, p
end

function on_words(classLayer::ClassificationLayer)
    if clf_exists(classLayer.id, "words")
        s, p = load_clf(classLayer.id, "words")
    else
        s, p = _states_classifier(classLayer.word_states)
        if classLayer.save_params
            save(classLayer.id, "words", s, p)
        end
    end
    s, p
end

function on_spikes(classLayer::ClassificationLayer, input::InputLayer)
    if clf_exists(classLayer.id, "spikes")
        s, p = load_clf(classLayer.id, "spikes")
    else
        interval = 100:100.:input.net.simulation_time
        spike_feats, spike_labels = LKD.spikes_to_features(classLayer.firing_times, input.transcriptions.phones, interval)
        s, p = LKD.MultiLogReg(spike_feats[1:input.weights_params.Ne,:], spike_labels)
        if classLayer.save_params
            save(input.id, "spikes", s, p)
        end
    end
    s, p
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

function kappa_score(y, y_hat)
    raw_acc = accuracy(y, y_hat)
    rand_acc = accuracy(y, shuffle(y_hat))
    (raw_acc - rand_acc) / (1 - rand_acc)
end

function featuresfromspikes(i::tt.InputLayer, o::tt.SNNOut, only_exc=true)
    fts, sgns = tt.LKD.spikes_to_features(o.firing_times, i.transcriptions.phones, 100:100.:i.net.simulation_time)
    if only_exc
        return fts[1:i.weights_params.Ne, :], sgns
    else
        fts, sgns
    end
end

function get_words_features(o::tt.SNNOut)
    X, _, labels = tt.LKD.states_to_features(o.word_states)
    X, labels
end

function get_spikes_features(o::tt.SNNOut)
    X, _, labels = tt.LKD.states_to_features(o.phone_states)
    X, labels
end

function words_classifier(id, out::tt.SNNOut)
    cl = tt.ClassificationLayer(id, out)
    s, θ = tt.on_words(cl)
    X, labels = tt.get_words_features(out)
    y, y_hat, n_classes = tt.predict(X, labels, θ)
    acc = tt.accuracy(y, y_hat)
    (accuracy=acc, y=y, y_hat=y_hat, labels=collect(Set(labels)))
end

function phones_classifier(id, out::tt.SNNOut)
    cl = tt.ClassificationLayer(id, out)
    s, θ = tt.on_phones(cl)
    X, labels = tt.get_spikes_features(out)
    y, y_hat, n_classes = tt.predict(X, labels, θ)
    acc = tt.accuracy(y, y_hat)
    (accuracy=acc, y=y, y_hat=y_hat, labels=collect(Set(labels)))    
end
