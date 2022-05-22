using Plots
using Statistics
using tt
using JLD2

function make_spike_raster(raw_spikes, id)
    scatter(
        tt.SpikeTimit.get_raster_data(
            vcat(
                raw_spikes[1:400],
                raw_spikes[4900:5000]
            )
        ),
        m=(1, :black, stroke(0)), leg = :none, yticks=false, xtickfontsize=5, labelfontsize=5, grid=false, 
        ylabel=(id == 1 ? "Neurons" : ""), xlabel="Time (ms)", xlims=:auto
    )
end

conf = tt.load_conf()
folder = joinpath(conf["experiments"], "data_exp_E_bae.jld2")
data = load(folder);
swords = ["the", "a", "water", "greasy"]
input_spikes = data["in_spikes"]
sim_triplet = data["sim_triplet"]
sim_voltage = data["sim_voltage"]

function mkchart(input_spikes, sim_data, swords)
    # scatter of input spikes
    scatter_input_spikes = [
        scatter(tt.SpikeTimit.get_raster_data(input_spikes[i][1]),m=(1, :black, stroke(0)), leg = :none, tickfontsize=5, labelfontsize=5, grid=false, xlabel="Time (s)", ylabel=(i == 1 ? "Neurons" : "")) 
            for i in 1:length(input_spikes)
    ]
    
    # scatter of network spikes after training
    net_scatter_cochlea_trip = [
            make_spike_raster(sim_data[i].net_spikes[10], i) for i in 1:length(swords)
    ]
    
    # firing rates
    firing_rates_trip = [
        plot(mean(sim_data[i].net_fr), leg=:none, c=:black, tickfontsize=5, labelfontsize=5, grid=:none, xlabel="Simulation timestep", ylabel=(i == 1 ? "Hz" : ""))
            for i in 1:length(swords)
    ]
    
    # all together
    t1 = ["$(swords[i]) ($i)" for j in 1:1, i in 1:4]
    t2 = ["" for j in 1:1, i in 5:12]
    titles = hcat(t1, t2)
    lyt = @layout [a b c d; e f g h; i j k l]
    mnp = reduce(hcat, [scatter_input_spikes, net_scatter_cochlea_trip, firing_rates_trip])
    return plot(mnp..., layout=lyt, size=(800, 400), title=titles, titleloc = :center, titlefont=7);
end

plt_triplet = mkchart(input_spikes, sim_triplet, swords);
plt_voltage = mkchart(input_spikes, sim_voltage, swords);
plt = plot(plt_triplet);
savefig(plt,"E_triplet_bae.pdf");
plt = plot(plt_voltage);
savefig(plt,"E_voltage_bae.pdf");


# For cochlear encoding
folder = joinpath(conf["experiments"], "data_exp_E_cochlea70.jld2")
data = load(folder);
swords = ["the", "a", "water", "greasy"]
input_spikes = data["in_spikes"]
sim_triplet = data["sim_triplet"]
sim_voltage = data["sim_voltage"]

plt_triplet = mkchart(input_spikes, sim_triplet, swords);
plt_voltage = mkchart(input_spikes, sim_voltage, swords);
plt = plot(plt_triplet);
savefig(plt,"E_triplet_cochlea70.pdf");
plt = plot(plt_voltage);
savefig(plt,"E_voltage_cochlea70.pdf");