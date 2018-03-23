__precompile__()
module NestedMill
using JSON 
using Adapt
using MLDataPattern

function paddedprint(io,s,padding)
	for _ in 1:padding
	    write(io, ' ')
	end
  write(io,s)
end

Bags = Vector{UnitRange{Int64}}


include("datanode.jl")
include("jsonschema.jl")
include("reflector.jl")
include("modelnode.jl")
include("aggregation.jl")


export DataNode
end
