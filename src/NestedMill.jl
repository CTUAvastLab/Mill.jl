__precompile__()
module NestedMill

Bags = Vector{UnitRange{Int64}}
include("ragged.jl")
include("joined.jl")
include("reflectjson.jl")

end
