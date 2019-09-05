@testset "testing simple matrix model" begin
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    x = ArrayNode(randn(Float32, 4, 5))
    m = reflectinmodel(x, layerbuilder)
    @test size(m(x).data) == (2, 5)
    @test typeof(m) <: ArrayModel
    @test eltype(m(x).data) == Float32
end

@testset "testing simple aggregation model" begin
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    x = BagNode(ArrayNode(randn(Float32, 4, 4)), [1:2, 3:4])
    m = reflectinmodel(x, layerbuilder)
    @test size(m(x).data) == (2, 2)
    @test typeof(m) <: BagModel
    @test eltype(m(x).data) == Float32
end

@testset "testing simple tuple models" begin
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    x = TreeNode((ArrayNode(randn(Float32, 3, 4)), ArrayNode(randn(Float32, 4, 4))))
    m = reflectinmodel(x, layerbuilder)

    @test eltype(m(x).data) == Float32
    @test size(m(x).data) == (2, 4)
    @test typeof(m) <: ProductModel
    @test typeof(m.ms[1]) <: ArrayModel
    @test typeof(m.ms[2]) <: ArrayModel
    x = TreeNode((BagNode(ArrayNode(randn(Float32, 3, 4)), [1:2, 3:4]),
                  BagNode(ArrayNode(randn(Float32, 4, 4)), [1:1, 2:4])))
    m = reflectinmodel(x, layerbuilder)
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
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    bn = BagNode(ArrayNode(randn(Float32, 2, 8)), [1:1, 2:2, 3:6, 7:8])
    x = BagNode(bn, [1:2, 3:4])
    m = reflectinmodel(x, layerbuilder)
    @test size(m(x).data) == (2, 2)
    @test typeof(m) <: BagModel
    @test typeof(m.im) <: BagModel
    @test typeof(m.im.im) <: ArrayModel
    @test typeof(m.im.bm) <: ArrayModel
    @test typeof(m.bm) <: ArrayModel
    @test eltype(m(x).data) == Float32
end
