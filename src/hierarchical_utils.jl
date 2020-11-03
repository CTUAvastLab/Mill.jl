import HierarchicalUtils: NodeType, LeafNode, InnerNode, noderepr, children

NodeType(::Type{<:Union{Missing, ArrayNode, ArrayModel, LazyNode}}) = LeafNode()
NodeType(::Type{<:AbstractNode}) = InnerNode()
NodeType(::Type{<:AbstractMillModel}) = InnerNode()
NodeType(::Type{<:LazyModel}) = InnerNode()

noderepr(::Missing) = "∅"
noderepr(n::LazyNode{N, Nothing}) where {N} = "LazyNode{$N} ∅"

noderepr(n::AbstractNode) = "$(nobs(n)) × " * _noderepr(n)
_noderepr(n::ArrayNode) = "ArrayNode($(summary(n.data)))"
_noderepr(n::BagNode) = "BagNode"
_noderepr(n::WeightedBagNode) = "WeightedNode (Σw = $(sum(n.weights)))"
_noderepr(n::AbstractProductNode) = "ProductNode"
_noderepr(n::LazyNode{N}) where {N} = "LazyNode{$N}"

noderepr(::T) where T <: Union{AbstractMillModel} = string(nameof(T))
noderepr(m::ArrayModel) = "ArrayModel($(m.m))"
noderepr(m::BagModel) = "BagModel ↦ $(m.a) ↦ $(noderepr(m.bm))"
noderepr(m::ProductModel) = "ProductModel ↦ $(noderepr((m.m)))"
noderepr(m::LazyModel{Name}) where {Name} = "Lazy$(Name)Model"

children(n::AbstractBagNode) = (n.data,)
children(n::BagModel) = (n.im,)
children(n::ProductNode) = n.data
children(n::ProductModel) = n.ms
children(n::LazyModel) = (n.m,)
