
using DataFrames
using StatsBase
using MLJLinearModels
using CairoMakie, ColorSchemes
using Plots
using Plots.PlotMeasures

function triosetup(id)
    e = tt.Experiment("trio", "")
    words = e.data.data[1]
    test_words = words[:test_words]
    train_words = words[:train_words]
    e = tt.Experiment("trio", "couples")
    train = e.data.data[1][:train]
    test = e.data.data[1][:test]

    trained_net_id = id
    v_trained_net_path = joinpath(tt.simsdir(), "VOL_"*trained_net_id)
    t_trained_net_path = joinpath(tt.simsdir(), "TRI_"*trained_net_id*"_minimal")

    vol_net = tt.load(v_trained_net_path)
    tri_net = tt.load(t_trained_net_path)
    il_vol = tt._loadinputlayer(id, tt.VoltageSTDP())
    il_tri = tt._loadinputlayer(id, tt.TripletSTDP())

    return (
        train,
        test,
        train_words,
        test_words,
        vol_net,
        tri_net,
        il_vol,
        il_tri
    )
end

"""returns words present in the dataframe and hence that can be used for testing"""
function get_usable_words(words::Vector{String}, df::DataFrame)
    usable = String[]
    for tw in words
        s = filter(:words => x -> any(tw in x), df).sentence
        if length(s) > 0
            push!(usable, tw)
        end
    end
    usable
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

function makeconfmat(data::Matrix{Float64}, labels, title; xlabelrotate=pi/2)
    n = length(labels)
	fig, ax, hm = CairoMakie.heatmap(
		#MLJLinearModels.softmax(data),
		data ./ sum(data, dims=2),
		colormap = ColorSchemes.bilbao.colors,
		figure = (backgroundcolor = :white,resolution = (1200, 1200),),
		axis = (aspect = 1, 
                ygridvisible=true,
                xgridvisible=true,
                yreversed=false,
				title=title,
				xlabel = "True", 
				ylabel = "Predicted", 
				titlesize = 22,
				xticklabelsize = 18,
				yticklabelsize = 18,
				ylabelsize = 20,
				xlabelsize = 20,
				xticks = (1:n, labels), 
				yticks = (1:n, labels),  
				xticklabelrotation = xlabelrotate)
	)
	Colorbar(fig[:, end+1], hm)
	fig
end

function weights_anime(trace_a::Matrix{Float64}, trace_b::Matrix{Float64}, N, labels::Tuple{String, String}, filename, post = true)
    colors = ColorScheme(distinguishable_colors(10, transform=protanopic))
    n = length(trace_a)
    @assert n == length(trace_b) == length(trace_c)
    @assert length(labels) == 2
    which = 2
    if !post 
        which = 1
    end
    data_a = [filter(x->x!=0, trace_a[i][which][1:N,1:N][:]) for i in 1:n]
    data_b = [filter(x->x!=0, trace_b[i][which][1:N,1:N][:]) for i in 1:n]
    la, lb = labels
    anim = Plots.Animation()
    anim = @Plots.animate for i ∈ 1:n
        Plots.histogram(
            linewidth=0,
            data_a[i], 
            color=colors[2], 
            alpha=0.9,
            xlabel="Weight", 
            ylabel="Count", 
            label=la
        )
        Plots.histogram!(
            linewidth=0,
            data_b[i], 
            color=colors[1], 
            alpha=0.8,
            xlabel="Weight", 
            ylabel="Count", 
            label=lb, 
            title="$(i/10)")
        Plots.plot!(
            size=(600,600),
            titlefontsize=10,
            labelfontsize=9,
            xlims=[1,6], 
            ylims=[0,800], 
            grid=false, 
            framestyle=:box
        )
    end
    Plots.gif(anim, joinpath(tt.plotsdir(), "$(filename).gif"), fps=5)
end

function weights_anime(trace_a::Matrix{Float64}, trace_b::Matrix{Float64}, trace_c::Matrix{Float64}, N, labels::Tuple{String, String, String}, filename, post = true)
    colors = ColorScheme(distinguishable_colors(10, transform=protanopic))
    n = length(trace_a)
    @assert n == length(trace_b) == length(trace_c)
    @assert length(labels) == 3
    which = 2
    if !post 
        which = 1
    end
    data_a = [filter(x->x!=0, trace_a[i][which][1:N,1:N][:]) for i in 1:n]
    data_b = [filter(x->x!=0, trace_b[i][which][1:N,1:N][:]) for i in 1:n]
    data_c = [filter(x->x!=0, trace_c[i][which][1:N,1:N][:]) for i in 1:n]
    la, lb, lc = labels
    anim = Plots.Animation()
    anim = @Plots.animate for i ∈ 1:n
        Plots.histogram(
            linewidth=0,
            data_a[i], 
            color=colors[2], 
            alpha=0.9,
            xlabel="Weight", 
            ylabel="Count", 
            label=la
        )
        Plots.histogram!(
            linewidth=0,
            data_b[i], 
            color=colors[3], 
            alpha=0.9,
            xlabel="Weight", 
            ylabel="Count", 
            label=lb
        )    
        Plots.histogram!(
            linewidth=0,
            data_c[i], 
            color=colors[1], 
            alpha=0.8,
            xlabel="Weight", 
            ylabel="Count", 
            label=lc, 
            title="$(i/10)")
        Plots.plot!(
            size=(600,600),
            titlefontsize=10,
            labelfontsize=9,
            xlims=[1,6], 
            ylims=[0,800], 
            grid=false, 
            framestyle=:box
        )
    end
    Plots.gif(anim, joinpath(tt.plotsdir(), "$(filename).gif"), fps=5)
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