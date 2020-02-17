module Mill
using Flux, MLDataPattern, SparseArrays, Statistics, Combinatorics, Zygote
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

# TODO delete
include("traversal_encoding.jl")

include("bagchain.jl")
export BagChain

include("replacein.jl")
export replacein

include("HierarchicalUtils/src/HierarchicalUtils.jl")
import .HierarchicalUtils: print_tree, head_string, tail_string, children_string, children, nchildren, NodeType, LeafNode, InnerNode

# TODO leafes by mely mit automaticky zadefinovane vlastnosti
# TODO aggregation function also MillModel?
Base.show(io::IO, ::MIME"text/plain", m::Union{AbstractNode, MillModel, AggregationFunction}) = print_tree(io, m, trunc_level=2, trav=false)
Base.show(io::IO, ::T) where T <: Union{AbstractNode, MillModel, AggregationFunction} = show(io, Base.typename(T))

# TODO
Base.show(io::IO, n::NGramMatrix) = (print(io, "NGramMatrix($(n.b), $(n.m))"); show(io, n.s))
Base.show(io::IO, ::MIME{Symbol("text/plain")}, n::NGramMatrix) = Base.show(io, n)

NodeType(::Type{<:Union{ArrayNode, ArrayModel, Missing}}) = LeafNode()
NodeType(::Type{<:AggregationFunction}) = LeafNode()
NodeType(::Type{<:AbstractNode}) = InnerNode()
NodeType(::Type{<:MillModel}) = InnerNode()

# TODO upravit Mill + jeho dokumentaci
# # TODO delete traversal
# TODO delete this and extract to a new package
import .HierarchicalUtils: COLORS, paddedprint

head_string(::Missing) = "∅"
tail_string(x::Missing) = ""
children_string(x::Missing) = []
children(x::Missing) = []
nchildren(::Missing) = 0

head_string(n::ArrayNode) = "ArrayNode$(size(n.data))"
tail_string(x::ArrayNode) = ""
children_string(x::ArrayNode) = []
children(x::ArrayNode) = []
nchildren(::ArrayNode) = 0

head_string(n::BagNode) = "BagNode$(size(n.data))"
head_string(n::BagNode) = "BagNode with $(length(n.bags)) bag(s)"
head_string(n::WeightedBagNode) = "WeightedNode with $(length(n.bags)) bag(s) and weights Σw = $(sum(n.weights))"
tail_string(::AbstractBagNode) = ""
children_string(::AbstractBagNode) = [""]
children(n::AbstractBagNode) = (n.data,)
nchildren(::AbstractBagNode) = 1

key_labels(data::NamedTuple) = 
key_labels(data) = ["" for _ in 1:length(data)]
head_string(n::TreeNode) = "TreeNode"
tail_string(n::TreeNode) = ""
children_string(n::TreeNode{<:NamedTuple}) = ["$k: " for k in keys(n.data)]
children_string(n::TreeNode) = ["" for k in keys(n.data)]
children(n::TreeNode) = n.data
nchildren(n::TreeNode) = length(n.data)

head_string(n::ArrayModel) = "ArrayModel($(n.m))"
tail_string(x::ArrayModel) = ""
children_string(x::ArrayModel) = []
children(x::ArrayModel) = []
nchildren(::ArrayModel) = 0

head_string(n::T) where T <: AggregationFunction = "$(T.name)($(length(n.C)))"
head_string(a::Aggregation{N}) where N = "⟨" * join(head_string(f) for f in a.fs ", ") * "⟩"
tail_string(::AggregationFunction) = ""
children_string(::AggregationFunction) = []
children(n::AggregationFunction) = []
nchildren(::AggregationFunction) = 0

head_string(n::BagModel) = "BagModel"
tail_string(::BagModel) = ""
children_string(::BagModel) = ["", "", ""]
children(n::BagModel) = (n.im, n.a, n.bm)
nchildren(::BagModel) = 3

head_string(n::ProductModel) = "ProductModel("
tail_string(n::ProductModel) = " ) ↦  $(n.m)"
children_string(n::ProductModel{<:NamedTuple}) = ["$k: " for k in keys(n.ms)]
children_string(n::ProductModel) = ["" for k in keys(n.ms)]
children(n::ProductModel) = n.ms
nchildren(n::ProductModel) = length(n.ms)

end
