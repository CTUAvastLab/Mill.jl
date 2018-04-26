__precompile__()
module Mill
using JSON 
using Adapt
using MLDataPattern

function paddedprint(io,s,padding)
	for _ in 1:padding
	    print(io, ' ')
	end
  print(io,s)
end

Bags = Vector{UnitRange{Int64}}


include("datanode.jl")
include("modelnode.jl")
include("aggregation.jl")


export DataNode
end
