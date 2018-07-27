using Revise
using Mill
using Base.Test
import Mill: DenseBuilder, ModelBuilder, ModelNode, DataNode, AggregationNode, reflectinmodel

layerbuilder(k) = Flux.Dense(k,2,NNlib.relu)

@testset "testing simple matrix model" begin
	x = MatrixNode(randn(4, 5))
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x)) == (2, 5)
	@test typeof(m) <: ModelNode
end

@testset "testing simple aggregation model" begin
	x = BagNode(MatrixNode(randn(4, 4)), [1:2, 3:4])
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x)) == (2, 2)
	@test typeof(m) <: AggregationNode
end

@testset "testing simple tuple models" begin
	x = TreeNode((MatrixNode(randn(3, 4)), MatrixNode(randn(4, 4))))
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x)) == (2, 4)
	@test typeof(m) <: ModelNode
	x = TreeNode((BagNode(MatrixNode(randn(3, 4)), [1:2, 3:4]), BagNode(MatrixNode(randn(4, 4)), [1:1, 2:4])))
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x)) == (2, 2)
	@test typeof(m) <: ModelNode
end

@testset "testing nested bag model" begin
	b = BagNode(MatrixNode(randn(2, 8)), [1:1, 2:2, 3:6, 7:8])
	x = BagNode(b, [1:2, 3:4])
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x)) == (2, 2)
end
