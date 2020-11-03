import HierarchicalUtils: NodeType, LeafNode, InnerNode, noderepr, children

NodeType(::Type{<:Union{Missing, ArrayNode, ArrayModel, LazyNode}}) = LeafNode()
NodeType(::Type{<:AbstractNode}) = InnerNode()
NodeType(::Type{<:AbstractMillModel}) = InnerNode()
NodeType(::Type{<:LazyModel}) = InnerNode()

noderepr(::Missing) = "∅"
noderepr(n::LazyNode{N, Nothing}) where {N} = "LazyNode{$N} ∅"

children(n::AbstractBagNode) = (n.data,)
children(n::BagModel) = (n.im,)
children(n::ProductNode) = n.data
children(n::ProductModel) = n.ms
children(n::LazyModel) = (n.m,)
