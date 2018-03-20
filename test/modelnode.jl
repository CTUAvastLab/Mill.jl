using Flux
import NestedMill: ModelNode, DataNode

x = DataNode([randn(3,4),randn(4,4)])
m = ModelNode((Dense(3,2),Chain(Dense(4,2),Dense(2,1))))

m(x)