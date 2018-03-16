__precompile__()
module NestedMill

Bags = Vector{UnitRange{Int64}}
include("datanode.jl")
# include("reflectjson.jl")


export DataNode
end
