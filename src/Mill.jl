__precompile__()
module Mill
using JSON
using Flux
using Adapt
using MLDataPattern

paddedprint(io,s,padding) = print(io, repeat(" ", padding), s)

const Bags = Vector{UnitRange{Int64}}
const VecOrRange = Union{UnitRange{Int},Vector{Int}}

include("bag_util.jl")
include("datanode.jl")
include("modelnode.jl")
include("aggregation.jl")

export AbstractNode, AbstractTreeNode, AbstractBagNode
export MatrixNode
export BagNode, WeightedBagNode
export TreeNode
export ModelNode, AggregationNode

end
