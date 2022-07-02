
using DataFrames
using StatsBase
using MLJLinearModels
using CairoMakie, ColorSchemes
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

function confmats4x4grid(t_cmw, t_cmp, v_cmw, v_cmp, words, phones, f = Figure(resolution = (1200, 1200)))
    function confmat(data, labels, f=Figure(); xlabelrotate=pi/2)
        n = length(labels)
        ax, hm = CairoMakie.heatmap(f[1,1],
            data ./ sum(data, dims=2),
            colormap = ColorSchemes.bilbao.colors,
            figure = (backgroundcolor = :white,resolution = (500, 500),),
            axis = (aspect = 1, 
                    ygridvisible=true,
                    xgridvisible=true,
                    yreversed=false,
            ))
            hidedecorations!(ax)
        #Colorbar(fig[:, end+1], hm)
        hm
    end
    

    ga = f[1, 1] = GridLayout()
    gb = f[1, 2] = GridLayout()
    gc = f[2, 1] = GridLayout()
    gd = f[2, 2] = GridLayout()    
    cm = confmat(t_cmw, words, ga)
    confmat(v_cmw, words, gb)
    confmat(t_cmp, phones, gc)
    confmat(v_cmp, phones, gd)
    colgap!(ga, 0)
    rowgap!(gb, 55)
    rowgap!(ga, 55)
    Colorbar(f[:, end+1], cm)
    f
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

function weights_anime(trace_a::Vector{Matrix{Float64}}, N, label, filename)
    colors = ColorScheme(distinguishable_colors(10, transform=protanopic))
    n = length(trace_a)
    t = collect(1:n)
    data_a = [filter(x->x!=0, trace_a[i][1:N,1:N][:]) for i in 1:n]
    anim = Plots.Animation()
    anim = @Plots.animate for i ∈ 1:n
        Plots.histogram(
            xlims=[1,12],
            ylims=[0,1e4],
            linewidth=0,
            data_a[i], 
            color=colors[2], 
            alpha=0.9,
            xlabel="Weight", 
            ylabel="Count", 
            title="$(t/100) s",
            label=label
        )
    end
    Plots.gif(anim, joinpath(tt.plotsdir(), "$(filename).gif"), fps=5)
end

function weights_anime(trace_a::Vector{Tuple{Float64, Array}}, N, label, filename, post = true)
    colors = ColorScheme(distinguishable_colors(10, transform=protanopic))
    n = length(trace_a)
    which = 2
    if !post 
        which = 1
    end
    t = map(x -> x[1], trace_a)
    trace_a = map(x -> x[2], trace_a)
    data_a = [filter(x->x!=0, trace_a[i][1:N,1:N][:]) for i in 1:n]
    anim = Plots.Animation()
    anim = @Plots.animate for i ∈ 1:n
        Plots.histogram(
            xlims=[1,12],
            ylims=[0,1e4],
            linewidth=0,
            data_a[i], 
            color=colors[2], 
            alpha=0.9,
            xlabel="Weight", 
            ylabel="Count", 
            title="$(t/100) s",
            label=label
        )
    end
    Plots.gif(anim, joinpath(tt.plotsdir(), "$(filename).gif"), fps=5)
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

function weights_hist(W, N, l = "")
    dw = filter(x->x!=0, W[10][2][1:N,1:N][:])
    Plots.histogram!(
        linewidth=1,
        dw, 
        color=:green, 
        #alpha=0.9,
        xlabel="Weight", 
        ylabel="Count", 
        label=l,
        bins=1.5:0.01:4.5
    )
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

function plotaccuracymatrices(acc_tri, acc_vol, f = Figure(), legend=true)
    # c = ColorScheme(distinguishable_colors(10, transform=protanopic))
    c = ColorSchemes.tab10
    color_tri = "#880000"
    color_vol = "#0000A5"
    col1 = color_tri
    col2 = color_vol
    x = collect(1:size(acc_vol, 1))
    plt = f[1,1]
    ax = Axis(plt, xlabel="Trial", ylabel="Accuracy", xticks=x, yticks=collect(0.5:0.1:1.0))
    lines!(plt,acc_tri[:,1], linestyle = :solid, linewidth = 2, color=col1, label="Words"); #red
    lines!(plt,acc_tri[:,2], linestyle = :dash, linewidth = 2, color=col1, label="Phones");
    Makie.scatter!(plt, x, acc_tri[:,1], color=col1, markersize = 10)
    Makie.scatter!(plt, x, acc_tri[:,2], color=col1, markersize = 10)
    lines!(plt,acc_vol[:,1], linestyle = :solid, linewidth = 2, color=col2, label="Words"); #purple
    lines!(plt,acc_vol[:,2], linestyle = :dash, linewidth = 2, color=col2, label="Phones");
    Makie.scatter!(plt, x, acc_vol[:,1], color=col2, markersize = 10)
    Makie.scatter!(plt, x, acc_vol[:,2], color=col2, markersize = 10)
    
    fl = Figure(resolution=(500,500))
    if legend
        group_tri = [[LineElement(color = col1, linestyle=:solid), MarkerElement(color=col1, marker = :circle, markersize = 10)], [LineElement(color = col1, linestyle=:dash), MarkerElement(color=col1, marker = :circle, markersize = 10)]]
        group_vol = [[LineElement(color = col2, linestyle=:solid), MarkerElement(color=col2, marker = :circle, markersize = 10)], [LineElement(color = col2, linestyle=:dash), MarkerElement(color=col2, marker = :circle, markersize = 10)]]
        Legend(fl[1, 1], [group_tri, group_vol], [["Words", "Phones"], ["Words", "Phones"]], ["Triplet" "Voltage"], nbanks=2)
    end
    Makie.ylims!(ax, 0.2, 0.7)
    f, fl
end