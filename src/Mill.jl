__precompile__()
module Mill
using JSON
using Adapt
using MLDataPattern

VecOrRange = Union{UnitRange{Int},V where V<:AbstractVector}

paddedprint(io,s,padding) = print(io, repeat(" ", padding), s)

const Bags = Vector{UnitRange{Int64}}

"""
		function length2bags(ls::Vector{Int})

		creates ranges of bags from a vector of bag lengths

		```juliadoc
		julia> length2bags([1,3,2])
		[1:1,2:4,5:6]
		```
"""
function length2bags(ls::Vector{Int})
	ls = vcat([0],cumsum(ls))
	bags = map(i -> i[1]+1:i[2],zip(ls[1:end-1],ls[2:end]))
	map(b -> isempty(b) ? (0:-1) : b,bags)
end

include("datanode.jl")
include("modelnode.jl")
include("aggregation.jl")


export DataNode, ModelNode, AggregationNode
end
