# import HierarchicalUtils: NodeType, noderepr, set_children, children
import HierarchicalUtils: NodeType, LeafNode, InnerNode, noderepr, children

NodeType(::Type{<:Union{Missing, ArrayNode, ArrayModel, LazyNode, LazyModel}}) = LeafNode()
NodeType(::Type{<:AbstractNode}) = InnerNode()
NodeType(::Type{<:AbstractMillModel}) = InnerNode()

noderepr(::T) where T <: Union{AbstractNode, AbstractMillModel} = "$(Base.nameof(T))"
noderepr(::Missing) = "∅"
noderepr(n::ArrayNode) = "ArrayNode$(size(n.data))"
noderepr(n::ArrayModel) = "ArrayModel($(n.m))"
noderepr(n::BagNode) = if ismissing(n.data) || nobs(n.data) == 0
    "BagNode with $(length(n.bags)) empty bag(s)"
else
    "BagNode with $(length(n.bags)) bag(s)"
end
noderepr(n::BagModel) = "BagModel ↦ $(repr("text/plain", n.a)) ↦ $(repr("text/plain", n.bm))"
noderepr(n::WeightedBagNode) = "WeightedNode with $(length(n.bags)) bag(s) and weights Σw = $(sum(n.weights))"
noderepr(n::AbstractProductNode) = "ProductNode"
noderepr(n::ProductModel) = "ProductModel ↦ $(noderepr(n.m))"
noderepr(n::MissingNode) = "Missing"
noderepr(n::MissingModel) = "Missing"
noderepr(n::LazyNode{N,D}) where {N,D} = "$(N) $(length(n.data)) items"
noderepr(n::LazyNode{N,D}) where {N,D<:Nothing} = "$(N) ∅"
noderepr(n::LazyModel{Name}) where {Name} = "Lazy$(Name)"

children(n::AbstractBagNode) = (n.data,)
children(n::BagModel) = (n.im,)
children(n::ProductNode) = n.data
children(n::ProductModel) = n.ms
children(n::MissingNode) = (n.data,)
children(n::MissingModel) = (n.data,)
