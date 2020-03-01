# TODO upravit Mill + jeho dokumentaci
# TODO MillModel -> AbstractMillModel
import HierarchicalUtils: NodeType, noderepr, childrenfield, children, childrenstring

NodeType(::Type{<:Union{ArrayNode, ArrayModel, BagNode{Missing}}}) = LeafNode()
NodeType(::Type{<:AbstractNode}) = InnerNode()
NodeType(::Type{<:AbstractBagNode}) = SingletonNode()
NodeType(::Type{<:BagModel}) = SingletonNode()
NodeType(::Type{<:MillModel}) = InnerNode()

noderepr(n::ArrayNode) = "ArrayNode$(size(n.data))"
noderepr(n::ArrayModel) = "ArrayModel($(n.m))"
noderepr(n::BagNode) = "BagNode with $(length(n.bags)) bag(s)"
noderepr(n::BagModel) = "BagModel ↦ $(repr("text/plain", n.a)) ↦ $(repr("text/plain", n.bm))"
noderepr(n::WeightedBagNode) = "WeightedNode with $(length(n.bags)) bag(s) and weights Σw = $(sum(n.weights))"
noderepr(n::AbstractTreeNode) = "TreeNode"
noderepr(n::ProductModel) = "ProductModel ↦ $(noderepr(n.m))"

childrenfield(::Type{<:Union{AbstractTreeNode, AbstractBagNode}}) = :data
childrenfield(::Type{BagModel}) = :im
childrenfield(::Type{ProductModel}) = :ms

children(n::AbstractBagNode) = (n.data,)
children(n::BagModel) = (n.im,)
children(n::TreeNode) = n.data
children(n::ProductModel) = n.ms

childrenstring(n::TreeNode{<:NamedTuple}) = ["$k: " for k in keys(n.data)]
childrenstring(n::ProductModel{<:NamedTuple}) = ["$k: " for k in keys(n.ms)]
