using Revise
using Mill
using Base.Test
import Mill: ChainModel, AggregationModel, JointModel, reflectinmodel

layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)

@testset "testing simple matrix model" begin
	x = ArrayNode(randn(4, 5))
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x).data) == (2, 5)
	@test typeof(m) <: ChainModel
end

@testset "testing simple aggregation model" begin
	x = BagNode(ArrayNode(randn(4, 4)), [1:2, 3:4])
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x).data) == (2, 2)
	@test typeof(m) <: AggregationModel
end

@testset "testing simple tuple models" begin
	x = TreeNode((ArrayNode(randn(3, 4)), ArrayNode(randn(4, 4))))
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x).data) == (2, 4)
	@test typeof(m) <: JointModel
	@test typeof(m.ms[1]) <: ChainModel
	@test typeof(m.ms[2]) <: ChainModel
	x = TreeNode((BagNode(ArrayNode(randn(3, 4)), [1:2, 3:4]), BagNode(ArrayNode(randn(4, 4)), [1:1, 2:4])))
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x).data) == (2, 2)
	@test typeof(m) <: JointModel
	@test typeof(m.ms[1]) <: AggregationModel
	@test typeof(m.ms[1].im) <: ChainModel
	@test typeof(m.ms[1].bm) <: ChainModel
	@test typeof(m.ms[2]) <: AggregationModel
	@test typeof(m.ms[2].im) <: ChainModel
	@test typeof(m.ms[2].bm) <: ChainModel
end

@testset "testing nested bag model" begin
	b = BagNode(ArrayNode(randn(2, 8)), [1:1, 2:2, 3:6, 7:8])
	x = BagNode(b, [1:2, 3:4])
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x).data) == (2, 2)
	@test typeof(m) <: AggregationModel
	@test typeof(m.im) <: AggregationModel
	@test typeof(m.im.im) <: ChainModel
	@test typeof(m.im.bm) <: ChainModel
	@test typeof(m.bm) <: ChainModel
end
