__precompile__()
module Mill
using JSON
using Flux
using Adapt
using MLDataPattern

paddedprint(io, s...; offset::Int=0, color::Int=15) = print_with_color(color, io, repeat(" ", offset), s...)

const Bags = Vector{UnitRange{Int64}}
const VecOrRange = Union{UnitRange{Int},Vector{Int}}
const MillFunction = Union{Flux.Dense, Flux.Chain, Function}

include("util.jl")
include("datanode.jl")
include("modelnode.jl")
include("aggregation.jl")

export AbstractNode, AbstractTreeNode, AbstractBagNode
export ArrayNode, BagNode, WeightedBagNode, TreeNode
export MillModel, ChainModel, AggregationModel, JointModel

end
