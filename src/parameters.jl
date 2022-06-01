using Parameters
ms = 1

# As in LKD paper
#= @with_kw struct TripletSTDP
    tau_plus::Float32 = 16.8ms      # time constant representing decrease rate of presynaptic detector r1
    tau_minus::Float32 = 33.7ms     # time constant representing decrease rate of postsynaptic detector o1
    tau_x = 101ms                   # time constant representing decrease rate of presynaptic detector r2
    tau_y = 125ms                   # time constant representing decrease rate of postsynaptic detector o2
    A_plus_2 = 7.5e-10
    A_plus_3 = 9.3e-3
    A_minus_2 = 7e-3
    A_minus_3 = 2.3e-4
end =#

# from Alessio
@with_kw struct TripletSTDP
    tau_plus::Float32 = 16.8ms      # time constant representing decrease rate of presynaptic detector r1
    tau_minus::Float32 = 33.7ms     # time constant representing decrease rate of postsynaptic detector o1
    tau_x = 1ms                   # time constant representing decrease rate of presynaptic detector r2
    tau_y = 114ms                   # time constant representing decrease rate of postsynaptic detector o2
    A_plus_2 = 0
    A_plus_3 = 6.5e-3
    A_minus_2 = 7.1e-3
    A_minus_3 = 0 
end

@with_kw struct VoltageSTDP 
    dummy = "This is a dummy struct to facilatate use of multiple dispatch. The actual params are in sim()."
end

STDP = Union{TripletSTDP, VoltageSTDP}