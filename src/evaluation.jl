
using DataFrames
using StatsBase
using MLJLinearModels
using Makie, ColorSchemes
using Plots
using Plots.PlotMeasures

function eval_words(id, out, type, makeplots=false)
    wc = tt.words_classifier(id, out)
    cmw = tt.compute_confmat(wc.y, wc.y_hat)
    if makeplots
        f = tt.makeconfmat(cmw, wc.labels, "Words ($type)")
        tt.saveplot(id*"cmw_($type).pdf", f);
    end
    wc, cmw
end

function eval_phones(id, out, type, makeplots=false)
    pc = tt.phones_classifier(id, out)
    cmp = tt.compute_confmat(pc.y, pc.y_hat)
    if makeplots
        f = tt.makeconfmat(cmp, pc.labels, "Phones ($type)")
        tt.saveplot(id*"cmp_($type).pdf", f);
    end
    pc, cmp
end

function evaluate(net_in::tt.InputLayer, net_out::tt.SNNOut, allinfo = false, makeplots = true)
    id = net_in.id
    out = net_out
    type = net_in.stdp isa tt.TripletSTDP ? "Triplet" : "Voltage"

    wc, cmw = eval_words(id, out, type, makeplots)
    pc, cmp = eval_phones(id, out, type, makeplots)

    if !allinfo
        (words=(wc.accuracy, wc.kappa), phones=(pc.accuracy, pc.kappa)) 
    else
        (words=wc.accuracy, phones=pc.accuracy, cmw=cmw, cmp=cmp, w_labels=wc.labels, p_labels=pc.labels)
    end
end

function evaluate(in_tri::tt.InputLayer, in_vol::tt.InputLayer, out_tri::tt.SNNOut, out_vol::tt.SNNOut)
    # read classifier weights
    exp_tws = tt.Experiment("TRI_"*in_tri.id, "class_word_states_minimal")
    exp_tps = tt.Experiment("TRI_"*in_tri.id, "class_phones_spikes_minimal")
    exp_vws = tt.Experiment("VOL_"*in_vol.id, "class_word_states")
    exp_vps = tt.Experiment("VOL_"*in_vol.id, "class_phones_spikes")
    theta_tws = exp_tws.data.data[end]
    theta_tps = exp_tps.data.data[end]
    theta_vws = exp_vws.data.data[end]
    theta_vps = exp_vps.data.data[end]

    # compute features from training data
    feats, signs_tri = tt.LKD.spikes_to_features(out_tri.firing_times, in_tri.transcriptions.phones, 100:100.:in_tri.net.simulation_time)
    X_tp = feats[1:in_tri.weights_params.Ne,:]
    feats, signs_vol = tt.LKD.spikes_to_features(out_vol.firing_times, in_vol.transcriptions.phones, 100:100.:in_vol.net.simulation_time)
    X_vp = feats[1:in_vol.weights_params.Ne,:]
    X_tw, _, labels_tw = tt.LKD.states_to_features(out_tri.word_states)
    X_vw, _, labels_vw = tt.LKD.states_to_features(out_vol.word_states)

    # predictions
    tri_words = predict(X_tw, labels_tw, theta_tws)
    tri_phones = predict(X_tp, signs_tri, theta_tps)
    vol_words = predict(X_vw, labels_vw, theta_vws)
    vol_phones = predict(X_vp, signs_vol, theta_vps)

    # confusion matrices
    cm_tw = compute_confmat(tri_words.y, tri_words.y_hat)
    cm_tp = compute_confmat(tri_phones.y, tri_phones.y_hat)
    cm_vw = compute_confmat(vol_words.y, vol_words.y_hat)
    cm_vp = compute_confmat(vol_phones.y, vol_phones.y_hat)

    # accuracy
    acc_tw = accuracy(tri_words.y, tri_words.y_hat)
    acc_tp = accuracy(tri_phones.y, tri_phones.y_hat)
    acc_vw = accuracy(vol_words.y, vol_words.y_hat)
    acc_vp = accuracy(vol_phones.y, vol_phones.y_hat)


    # Figures
    lbl_w = collect(Set(labels_vw))
    lbl_p = collect(Set(signs_vol))

    fig_tw = makeconfmat(cm_tw, lbl_w, "TripletSTDP"; xlabelrotate = pi/3)
    fig_tp = makeconfmat(cm_tp, lbl_p, "TripletSTDP")
    fig_vw = makeconfmat(cm_vw, lbl_w, "VoltageSTDP"; xlabelrotate = pi/3)
    fig_vp = makeconfmat(cm_vp, lbl_p, "VoltageSTDP")

    tt.saveplot(in_tri.id*"cm_tw.pdf", fig_tw)
    tt.saveplot(in_tri.id*"cm_tp.pdf", fig_tp)
    tt.saveplot(in_tri.id*"cm_vw.pdf", fig_vw)
    tt.saveplot(in_tri.id*"cm_vp.pdf", fig_vp)

    exp_cm_tw = tt.Experiment(in_tri.id, "cm_tw")
    exp_cm_tp = tt.Experiment(in_tri.id, "cm_tp")
    exp_cm_vw = tt.Experiment(in_tri.id, "cm_vw")
    exp_cm_vp = tt.Experiment(in_tri.id, "cm_vp")

    tt.save(Dict(:cm => cm_tw, :labels => lbl_w), exp_cm_tw)
    tt.save(Dict(:cm => cm_tp, :labels => lbl_p), exp_cm_tp)
    tt.save(Dict(:cm => cm_vw, :labels => lbl_w), exp_cm_vw)
    tt.save(Dict(:cm => cm_vp, :labels => lbl_p), exp_cm_vp)

    df_accuracy = DataFrame(:model=>String[], :accuracy=>Float64[])
    push!(df_accuracy, ("vol_w", acc_vw))
    push!(df_accuracy, ("vol_p", acc_vp))
    push!(df_accuracy, ("tri_w", acc_tw))
    push!(df_accuracy, ("tri_p", acc_tp))

    # plot accuracy
    f = Figure()
    ax = Axis(
        f[1,1], title="Classification accuracy",
        xticks=(1:2, ["Voltage", "Triplet"]),
        yautolimitmargin=(0.05, 0.1)
        )
    height = df_accuracy[!,2]
    bar_labels = repeat(["Words", "Phones"], 2) .* " (" .* string.(round.(df_accuracy[!,2]; digits=3)) .* ")"
    barplot!(
        ax,
        [1,1,2,2],
        height, 
        dodge=[1,2,1,2], 
        color=[1,2,1,2], 
        bar_labels=bar_labels,
    )
    hidedecorations!(ax, grid=true, ticklabels = false, ticks = false, label=false);
    f
    tt.saveplot(in_tri.id*"accuracy.pdf", f)
    
end
