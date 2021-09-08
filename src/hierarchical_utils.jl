import HierarchicalUtils: NodeType, LeafNode, InnerNode, children

@nospecialize

NodeType(::Type{<:Union{Missing, ArrayNode, ArrayModel, LazyNode}}) = LeafNode()
NodeType(::Type{<:AbstractMillNode}) = InnerNode()
NodeType(::Type{<:AbstractMillModel}) = InnerNode()
NodeType(::Type{<:LazyModel}) = InnerNode()

children(n::AbstractBagNode) = (n.data,)
children(n::BagModel) = (n.im,)
children(n::ProductNode) = n.data
children(n::ProductModel) = n.ms
children(n::LazyModel) = (n.m,)

@specialize
