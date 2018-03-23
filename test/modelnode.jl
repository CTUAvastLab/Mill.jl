using Revise
using Flux
using NestedMill
import NestedMill: ModelNode, DataNode, AggregationNode, reflectinmodel


x = DataNode(randn(4,4),[1:2,3:4])
layerbuilder(k) = Dense(k,10,relu),10

m = reflectinmodel(x,layerbuilder)[1]
m(x)


x = DataNode((randn(3,4),randn(4,4)))
m = reflectinmodel(x,layerbuilder)[1]
m(x)

x = DataNode((DataNode(randn(7,3)),DataNode(randn(4,3)))))
m = reflectinmodel(x,layerbuilder)[1]
m(x)

# x = DataNode((DataNode(randn(3,4)),DataNode(randn(4,4))))
# m = ModelNode((Dense(3,2),Chain(Dense(4,2),Dense(2,1))))
# m(x)


# x = DataNode(randn(4,4))
# m = ModelNode(Dense(4,2))
# m(x)


# x = DataNode(randn(4,4),[1:2,3:4])
#  = DataNode(nothing,[0:-1])
# m = AggregationNode(Dense(4,2),NestedMill.segmented_mean)


