
replacein(x, oldnode, newnode) = x

replacein(x::ArrayModel, oldnode, newnode) = x == oldnode ? newnode : ArrayModel(replacein(x.m, oldnode, newnode))
replacein(x::ArrayNode, oldnode, newnode) = x == oldnode ? newnode : ArrayNode(replacein(x.m, oldnode, newnode), x.metadata)

replacein(x::BagModel, oldnode, newnode) = x == oldnode ? newnode : BagModel(replacein(x.im, oldnode, newnode), replacein(x.a, oldnode, newnode), replacein(x.bm, oldnode, newnode))
replacein(x::BagNode, oldnode, newnode) = x == oldnode ? newnode : BagNode(replacein(x.data, oldnode, newnode), x.bags, x.metadata)


replacein(x::TreeNode, oldnode, newnode) = x == oldnode ? newnode : TreeNode(tuple([replacein(m, oldnode, newnode) for m in x.data]...), x.metadata)
replacein(x::ProductModel, oldnode, newnode) = x == oldnode ? newnode : ProductModel(tuple([replacein(m, oldnode, newnode) for m in x.ms]...), replacein(x.m, oldnode, newnode))