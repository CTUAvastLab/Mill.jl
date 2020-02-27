module Mill

using Flux, MLDataPattern, SparseArrays, Statistics, Combinatorics, Zygote, HierarchicalUtils
using Zygote: @adjoint
import Base.reduce

MLDataPattern.nobs(::Missing) = nothing

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

include("conv.jl")
export bagconv, BagConv

include("bagchain.jl")
export BagChain

include("replacein.jl")
export replacein

# TODO upravit Mill + jeho dokumentaci
import HierarchicalUtils: NodeType, treerepr, childrenfield, child

Base.show(io::IO, ::MIME"text/plain", m::Union{AbstractNode, MillModel, AggregationFunction}) = print_tree(io, m, trunc_level=2, trav=false)
Base.show(io::IO, ::T) where T <: Union{AbstractNode, MillModel, AggregationFunction} = show(io, Base.typename(T))
Base.show(io::IO, n::NGramMatrix) = (print(io, "NGramMatrix($(n.b), $(n.m))"); show(io, n.s))
Base.show(io::IO, ::MIME"text/plain", n::NGramMatrix) = Base.show(io, n)

NodeType(::Type{<:Union{ArrayNode, ArrayModel, Missing}}) = LeafNode()
NodeType(::Type{<:AggregationFunction}) = LeafNode()
NodeType(::Type{<:AbstractNode}) = InnerNode()
NodeType(::Type{<:MillModel}) = InnerNode()

treerepr(::Missing) = "∅"
treerepr(n::ArrayNode) = "ArrayNode$(size(n.data))"
treerepr(n::BagNode) = "BagNode$(size(n.data))"
treerepr(n::BagNode) = "BagNode with $(length(n.bags)) bag(s)"
treerepr(n::WeightedBagNode) = "WeightedNode with $(length(n.bags)) bag(s) and weights Σw = $(sum(n.weights))"

children_string(::AbstractBagNode) = [""]
children(n::AbstractBagNode) = (n.data,)
nchildren(::AbstractBagNode) = 1

key_labels(data::NamedTuple) = 
key_labels(data) = ["" for _ in 1:length(data)]
treerepr(n::TreeNode) = "TreeNode"
# tail_string(n::TreeNode) = ""
children_string(n::TreeNode{<:NamedTuple}) = ["$k: " for k in keys(n.data)]
children_string(n::TreeNode) = ["" for k in keys(n.data)]
children(n::TreeNode) = n.data
nchildren(n::TreeNode) = length(n.data)

treerepr(n::ArrayModel) = "ArrayModel($(n.m))"
# tail_string(x::ArrayModel) = ""

treerepr(n::T) where T <: AggregationFunction = "$(T.name)($(length(n.C)))"
treerepr(a::Aggregation{N}) where N = "⟨" * join(treerepr(f) for f in a.fs ", ") * "⟩"
# tail_string(::AggregationFunction) = ""

treerepr(n::BagModel) = "BagModel"
# tail_string(::BagModel) = ""
children_string(::BagModel) = ["", "", ""]
children(n::BagModel) = (n.im, n.a, n.bm)
nchildren(::BagModel) = 3

treerepr(n::ProductModel) = "ProductModel ↦ $(treerepr(n.m))"
# tail_string(n::ProductModel) = " ) ↦  $(n.m)"
children_string(n::ProductModel{<:NamedTuple}) = ["$k: " for k in keys(n.ms)]
children_string(n::ProductModel) = ["" for k in keys(n.ms)]
children(n::ProductModel) = n.ms
nchildren(n::ProductModel) = length(n.ms)

end
