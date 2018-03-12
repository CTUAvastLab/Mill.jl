__precompile__()
module NestedMill

Bags = Vector{UnitRange{Int64}}
include("ragged.jl")
include("reflectjson.jl")

end
