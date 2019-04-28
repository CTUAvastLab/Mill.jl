using Mill

root_node = BagNode(BagNode(ArrayNode(rand(3,4)),[1:2, 3:4]), [1:2])
root_model = Mill.reflectinmodel(root_node, d -> Dense(d, 4), d -> SegmentedMeanMax())
a, m = root_node.data, root_model.im
ϕ = (x...) -> Mill.onoff(x..., 1)

explain(root_node, root_model, (x...) -> Mill.onoff(x..., 1), explaining_fun)

