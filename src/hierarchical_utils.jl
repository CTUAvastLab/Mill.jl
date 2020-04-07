import HierarchicalUtils: NodeType, noderepr, childrenfields, children

NodeType(::Type{<:Union{Missing, ArrayNode, ArrayModel}}) = LeafNode()
NodeType(::Type{<:AbstractNode}) = InnerNode()
NodeType(::Type{<:AbstractBagNode}) = SingletonNode()
NodeType(::Type{<:BagModel}) = SingletonNode()
NodeType(::Type{<:AbstractMillModel}) = InnerNode()

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

childrenfields(::Type{<:Union{AbstractProductNode, AbstractBagNode}}) = (:data,)
childrenfields(::Type{BagModel}) = (:im,)
childrenfields(::Type{ProductModel}) = (:ms,)

children(n::AbstractBagNode) = (n.data,)
children(n::BagModel) = (n.im,)
children(n::ProductNode) = n.data
children(n::ProductModel) = n.ms
