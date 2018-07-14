
import Mill: ModelNode, DataNode, AggregationNode, reflectinmodel

layerbuilder(k) = Flux.Dense(k,2,NNlib.relu)

@testset "testing simple aggregation model" begin
		x = DataNode(randn(4,4),[1:2,3:4])
		m = reflectinmodel(x,layerbuilder)[1]
		@test size(m(x)) == (2,2)
		@test typeof(m) <: AggregationNode
end

@testset "testing simple tuple model" begin
		x = DataNode((randn(3,4),randn(4,4)))
		m = reflectinmodel(x,layerbuilder)[1]
		@test size(m(x)) == (2,4)
		x = DataNode((DataNode(randn(3,4)),DataNode(randn(4,4))))
		m = reflectinmodel(x,layerbuilder)[1]
		@test size(m(x)) == (2,4)
		x = DataNode((randn(3,4),DataNode(randn(4,4))))
		m = reflectinmodel(x,layerbuilder)[1]
		@test size(m(x)) == (2,4)
		x = DataNode((DataNode(randn(3,4)),randn(4,4)))
		m = reflectinmodel(x,layerbuilder)[1]
		@test size(m(x)) == (2,4)
		x = DataNode((DataNode(randn(3,4),[1:2,3:4]),DataNode(randn(4,4),[1:1,2:4])))
		m = reflectinmodel(x,layerbuilder)[1]
		@test size(m(x)) == (2,2)
end
