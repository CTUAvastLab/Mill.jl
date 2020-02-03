using Mill, Test, Flux
using Mill: HiddenLayerModel, mapactivations
@testset "activations of simple matrix model" begin
    x = ArrayNode(randn(2, 4))
    m = ArrayModel(Chain(Dense(2,2,leakyrelu), Dense(2,2)))
    hm, o = HiddenLayerModel(m, x, 3)
    os = Flux.activations(m.m, x.data)

    hx, mx = mapactivations(hm, x, m)

    @test o.data ≈ m(x).data
    @test mx.data ≈ m(x).data
    @test hx.data ≈ hm.m[1](os[1]) + hm.m[2](os[2])
end

@testset "testing simple aggregation model" begin
    x = BagNode(ArrayNode(randn(2, 4)), [1:2,3:4])
    m = BagModel(
        ArrayModel(Chain(Dense(2,2,leakyrelu), Dense(2,2))),
        SegmentedMeanMax(2),
        ArrayModel(Chain(Dense(4,2,leakyrelu), Dense(2,2))))

    hm, o = HiddenLayerModel(m, x, 3)
    hx, mx = mapactivations(hm, x, m)

    @test size(hx.data) == (3,2)
    @test mx.data ≈ m(x).data


    x = BagNode(missing, [0:-1, 0:-1])
    hx, mx = mapactivations(hm, x, m)
    @test size(hx.data) == (3,2)
    @test mx.data ≈ m(x).data
end

@testset "testing simple tuple models" begin
    x = TreeNode((a = ArrayNode(randn(2,2)),
        b = ArrayNode(randn(3,2))))
    m = ProductModel((
        a = ArrayModel(Chain(Dense(2,2,leakyrelu), Dense(2,2))),
        b = ArrayModel(Chain(Dense(3,3,leakyrelu), Dense(3,3)))))

    hm, o = HiddenLayerModel(m, x, 3)
    hx, mx = mapactivations(hm, x, m)
    @test size(hx.data) == (3,2)
    @test mx.data ≈ m(x).data

    @test hx.data ≈ mapactivations(hm.ms.a, x.data.a, m.ms.a)[1].data + 
    mapactivations(hm.ms.b, x.data.b, m.ms.b)[1].data + 
    mapactivations(hm.m, vcat(m.ms.a(x.data.a), m.ms.b(x.data.b)), m.m)[1].data
end
