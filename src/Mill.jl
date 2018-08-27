__precompile__()
module Mill
using JSON, Flux, Adapt, MLDataPattern, SparseArrays, Statistics

paddedprint(io, s...; offset::Int=0, color::Int=15) = printstyled(io, repeat(" ",offset), s..., color = color)

const Bags = Vector{UnitRange{Int64}}
const VecOrRange = Union{UnitRange{Int},AbstractVector{Int}}
const MillFunction = Union{Flux.Dense, Flux.Chain, Function}

include("util.jl")
include("datanode.jl")
include("modelnode.jl")
include("aggregation.jl")

export AbstractNode, AbstractTreeNode, AbstractBagNode
export ArrayNode, BagNode, WeightedBagNode, TreeNode
export MillModel, ArrayModel, BagModel, ProductModel

end
