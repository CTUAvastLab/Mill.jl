__precompile__()
module NestedMill

Bags = Vector{UnitRange{Int64}}
include("datanode.jl")
# include("jsonschema.jl")
include("reflector.jl")


export DataNode
end
