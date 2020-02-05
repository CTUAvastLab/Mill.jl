module Mill
using Flux, MLDataPattern, SparseArrays, Statistics, Combinatorics, Zygote
using Zygote: @adjoint
import Base.reduce

const COLORS = [:blue, :red, :green, :yellow, :cyan, :magenta]

MLDataPattern.nobs(::Missing) = nothing

function paddedprint(io, s...; color=:default, pad=[])
    for (c, p) in pad
        printstyled(io, p, color=c)
    end
    printstyled(io, s..., color=color)
end

key_labels(data::NamedTuple) = ["$k: " for k in keys(data)]
key_labels(data) = ["" for _ in 1:length(data)]

const VecOrRange = Union{UnitRange{Int},AbstractVector{Int}}

"""
	catobs(xs...)

	concatenates all observations from all xs together
"""
function catobs end;

include("bags.jl")
export AlignedBags, ScatteredBags

include("util.jl")
include("threadfuns.jl")

include("datanodes/datanode.jl")
export AbstractNode, AbstractTreeNode, AbstractBagNode
export ArrayNode, BagNode, WeightedBagNode, TreeNode
export NGramMatrix, NGramIterator
export catobs, removeinstances

include("aggregations/aggregation.jl")
# agg. types exported in aggregation.jl
export Aggregation

include("modelnodes/modelnode.jl")
export MillModel, ArrayModel, BagModel, ProductModel
export reflectinmodel

include("traversal_encoding.jl")
export show_traversal, encode_traversal

include("conv.jl")
export bagconv, BagConv

include("bagchain.jl")
export BagChain

include("replacein.jl")
export replacein

end
