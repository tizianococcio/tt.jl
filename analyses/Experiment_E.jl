using FileIO
using Plots
using Statistics
using YAML
using tt
#using SpikeTimit

conf = YAML.load_file(joinpath(@__DIR__, "../conf/paths.yml"))
folder = joinpath(conf["experiments"], "E_bae.jld2")
data = load(folder);

swords = ["the", "a", "water", "greasy"]
my_xlims = [0, 3500]
inputs_cochlea = data["in_spikes"]
sim_triplet = data["sim_triplet"]
sim_voltage = data["sim_voltage"]

# scatter of input spikes
scatter_cochlea = [
    scatter(tt.SpikeTimit.get_raster_data(inputs_cochlea[i][1]),m=(2, :black, stroke(0)), leg = :none, tickfontsize=5, labelfontsize=5, grid=false, xlabel="Time (ms)", ylabel=(i == 1 ? "Neurons" : ""), xlims=my_xlims/10000) 
        for i in 1:length(inputs_cochlea)
]

# scatter of network spikes after training
net_scatter_cochlea_trip = [
    scatter(tt.SpikeTimit.get_raster_data(sim_voltage[i].net_spikes[10]), m=(1, :black, stroke(0)), leg = :none, yticks=false, xtickfontsize=5, labelfontsize=5, xlabel="Time (ms)", ylabel=(i == 1 ? "Neurons" : ""), grid=false, xlims=my_xlims/10)
        for i in 1:length(swords)
]

# firing rates
firing_rates_trip = [
    plot(mean(sim_voltage[i].net_fr), leg=false, c=:black, tickfontsize=5, labelfontsize=5, 
    xlabel="Time (ms) dt", ylabel=(i == 1 ? "Hz" : ""), grid=false, xlims=my_xlims) 
        for i in 1:length(swords)
]

# all together
t1 = ["$(swords[i]) ($i)" for j in 1:1, i in 1:4]
t2 = ["" for j in 1:1, i in 5:12]
titles = hcat(t1, t2)
lyt = @layout [a b c d; e f g h; i j k l]
mnp = reduce(hcat, [scatter_cochlea, net_scatter_cochlea_trip, firing_rates_trip])
plot(mnp..., layout=lyt, size=(800, 400), title=titles, titleloc = :center, titlefont=7)
#savefig("triplet_cochlea70_10iter.pdf");
