using LKD
using DataFrames
using Dates

@with_kw mutable struct Experiment
    name::String
    type::String=nothing
    df::DataFrame # full dataframe
    data::DataFrame # filtered dataframe with type
end

function Experiment(name::String, type::String)
    f = joinpath(tt.processeddatadir(), string(name)*".jld2")
    df = DataFrame(type=String[], data=Any[], timestamp=DateTime[])
    fdf = DataFrame(type=String[], data=Any[], timestamp=DateTime[])
    if isfile(f)
        data = jldopen(f, "r")
        df = data["data"]
        fdf = filter(:type => x->x==type, df)
        close(data)
    else
        data = jldopen(f, "w")
        data["data"] = df
        close(data)
    end
    Experiment(name, type, df, fdf)
end

function save(datas, e::tt.Experiment)
    # create temp copy for safety
    f = joinpath(tt.processeddatadir(), string(e.name)*".jld2")
    f_temp = joinpath(tt.processeddatadir(), string(e.name)*"_old.jld2")
    mv(f, f_temp)
    try
        data = jldopen(f, "w")
        push!(e.df, Dict(:type=>e.type, :data=>datas, :timestamp=>now()))
        data["data"] = e.df
        close(data)        
    catch er
        close(data)
    end
    # delete temp copy
    rm(f_temp)
end

function delete(rows::Vector{Int}, e::tt.Experiment)
    delete!(e.df, rows)
    # create temp copy for safety
    f = joinpath(tt.processeddatadir(), string(e.name)*".jld2")
    f_temp = joinpath(tt.processeddatadir(), string(e.name)*"_old.jld2")
    mv(f, f_temp)
    try
        data = jldopen(f, "w")
        data["data"] = e.df
        close(data)        
    catch er
        close(data)
    end
    # delete temp copy
    rm(f_temp)
end