using Mill, Test
using Mill: ArrayModel, BagModel, ProductModel, reflectinmodel

layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)

@testset "testing simple matrix model" begin
	x = ArrayNode(randn(4, 5))
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x).data) == (2, 5)
	@test typeof(m) <: ArrayModel
end

@testset "testing simple aggregation model" begin
	x = BagNode(ArrayNode(randn(4, 4)), [1:2, 3:4])
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x).data) == (2, 2)
	@test typeof(m) <: BagModel
end

@testset "testing simple tuple models" begin
	x = TreeNode((ArrayNode(randn(3, 4)), ArrayNode(randn(4, 4))))
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x).data) == (2, 4)
	@test typeof(m) <: ProductModel
	@test typeof(m.ms[1]) <: ArrayModel
	@test typeof(m.ms[2]) <: ArrayModel
	x = TreeNode((BagNode(ArrayNode(randn(3, 4)), [1:2, 3:4]), BagNode(ArrayNode(randn(4, 4)), [1:1, 2:4])))
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x).data) == (2, 2)
	@test typeof(m) <: ProductModel
	@test typeof(m.ms[1]) <: BagModel
	@test typeof(m.ms[1].im) <: ArrayModel
	@test typeof(m.ms[1].bm) <: ArrayModel
	@test typeof(m.ms[2]) <: BagModel
	@test typeof(m.ms[2].im) <: ArrayModel
	@test typeof(m.ms[2].bm) <: ArrayModel
end

@testset "testing nested bag model" begin
	b = BagNode(ArrayNode(randn(2, 8)), [1:1, 2:2, 3:6, 7:8])
	x = BagNode(b, [1:2, 3:4])
	m = reflectinmodel(x, layerbuilder)[1]
	@test size(m(x).data) == (2, 2)
	@test typeof(m) <: BagModel
	@test typeof(m.im) <: BagModel
	@test typeof(m.im.im) <: ArrayModel
	@test typeof(m.im.bm) <: ArrayModel
	@test typeof(m.bm) <: ArrayModel
end
