# use only activations without "kinks" for numerical gradient checking
# see e.g. https://stackoverflow.com/questions/40623512/how-to-check-relu-gradient
const ACTIVATIONS = [identity, σ, swish, softplus, logcosh, mish, tanhshrink, lisht]

@testset "testing simple matrix model" begin
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    x = ArrayNode(randn(Float32, 4, 5))
    m = reflectinmodel(x, layerbuilder)
    @test size(m(x).data) == (2, 5)
    @test m isa ArrayModel
    @test eltype(m(x).data) == Float32
    @inferred m(x)
end

@testset "testing simple aggregation model" begin
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    x = BagNode(ArrayNode(randn(Float32, 4, 4)), [1:2, 3:4])
    m = reflectinmodel(x, layerbuilder)
    @test size(m(x).data) == (2, 2)
    @test m isa BagModel
    @test eltype(m(x).data) == Float32
    @inferred m(x)
end

@testset "testing simple tuple models" begin
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    x = ProductNode((a=ArrayNode(randn(Float32, 3, 4)), b=ArrayNode(randn(Float32, 4, 4))))
    m = reflectinmodel(x, layerbuilder)
    @test eltype(m(x).data) == Float32
    @test size(m(x).data) == (2, 4)
    @test m isa ProductModel
    @test m.ms[:a] isa ArrayModel
    @test m.ms[:b] isa ArrayModel
    @inferred m(x)

    x = ProductNode((BagNode(ArrayNode(randn(Float32, 3, 4)), [1:2, 3:4]),
                     BagNode(ArrayNode(randn(Float32, 4, 4)), [1:1, 2:4])))
    m = reflectinmodel(x, layerbuilder)
    @test size(m(x).data) == (2, 2)
    @test m isa ProductModel
    @test m.ms[1] isa BagModel
    @test m.ms[1].im isa ArrayModel
    @test m.ms[1].bm isa ArrayModel
    @test m.ms[2] isa BagModel
    @test m.ms[2].im isa ArrayModel
    @test m.ms[2].bm isa ArrayModel
    @inferred m(x)
end

@testset "testing nested bag model" begin
    bn = BagNode(ArrayNode(randn(Float32, 2, 8)), [1:1, 2:2, 3:6, 7:8])
    x = BagNode(bn, [1:2, 3:4])
    m = reflectinmodel(x, d -> Flux.Dense(d, 2))
    @test size(m(x).data) == (2, 2)
    @test m isa BagModel
    @test m.im isa BagModel
    @test m.im.im isa ArrayModel
    @test m.im.bm isa ArrayModel
    @test m.bm isa ArrayModel
    @test eltype(m(x).data) == Float32
    @inferred m(x)

    a = BagNode(BagNode(ArrayNode(randn(2,2)),[1:2]),[1:1])
    b = BagNode(missing,[0:-1])
    c = BagNode(a.data[1:0], [0:-1])
    m = reflectinmodel(a)
    abc = catobs(a, b, c)
    bca = catobs(b, c, a)
    ma = m(a).data
    mb = m(b).data
    mc = m(c).data
    mabc = m(abc).data
    mbca = m(bca).data
    @test mb ≈ mc
    @test mabc[:,1] ≈ ma
    @test mabc[:,2] ≈ mb
    @test mabc[:,3] ≈ mc
    @test mbca[:,1] ≈ mb
    @test mbca[:,2] ≈ mc
    @test mbca[:,3] ≈ ma
    @inferred m(a)
    @inferred m(b)
    @inferred m(c)
end

@testset "reflectinmodel for missing" begin
    f1 = x -> reflectinmodel(x)
    f2 = x -> reflectinmodel(x, d -> Flux.Chain(Dense(d, 10), Dense(10, 10)))

    x1 = maybehot(2, 1:3) |> ArrayNode
    x2 = maybehot(missing, 1:3) |> ArrayNode
    @test f1(x1).m isa PostImputingDense
    @test f1(x2).m isa PostImputingDense
    @test f2(x1).m[1] isa PostImputingDense
    @test f2(x2).m[1] isa PostImputingDense
    for f in [f1, f2], x in [x1, x2]
        @inferred f(x)(x)
    end

    x1 = maybehotbatch([1, 2, 3], 1:3) |> ArrayNode
    x2 = maybehotbatch([1, 2, missing], 1:3) |> ArrayNode
    x3 = maybehotbatch(fill(missing, 3), 1:3) |> ArrayNode
    @test f1(x1).m isa PostImputingDense
    @test f1(x2).m isa PostImputingDense
    @test f1(x3).m isa PostImputingDense
    @test f2(x1).m[1] isa PostImputingDense
    @test f2(x2).m[1] isa PostImputingDense
    @test f2(x3).m[1] isa PostImputingDense
    for f in [f1, f2], x in [x1, x2, x3]
        @inferred f(x)(x)
    end

    x1 = NGramMatrix(["a", "b", "c"]) |> ArrayNode
    x2 = NGramMatrix(["a", missing, "c"]) |> ArrayNode
    x3 = NGramMatrix(fill(missing, 3)) |> ArrayNode
    @test f1(x1).m isa Flux.Dense
    @test f1(x2).m isa PostImputingDense
    @test f1(x3).m isa PostImputingDense
    @test f2(x1).m[1] isa Flux.Dense
    @test f2(x2).m[1] isa PostImputingDense
    @test f2(x3).m[1] isa PostImputingDense
    for f in [f1, f2], x in [x1, x2, x3]
        @inferred f(x)(x)
    end

    x1 = rand([1, 2], 3, 3) |> ArrayNode
    x2 = rand([1, 2, missing], 3, 3) |> ArrayNode
    x3 = fill(missing, 3, 3) |> ArrayNode
    @test f1(x1).m isa Flux.Dense
    @test f1(x2).m isa PreImputingDense
    @test f1(x3).m isa PreImputingDense
    @test f2(x1).m[1] isa Flux.Dense
    @test f2(x2).m[1] isa PreImputingDense
    @test f2(x3).m[1] isa PreImputingDense
    for f in [f1, f2], x in [x1, x2, x3]
        @inferred f(x)(x)
    end
end

# pn.m should be identity for any product node pn with a single key
@testset "single key dictionary reflect in model" begin
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    fsm = Dict("" => layerbuilder)

    x1 = (ArrayNode(randn(Float32, 3, 4)),) |> ProductNode
    x2 = (a = ArrayNode(randn(Float32, 3, 4)),) |> ProductNode
    x3 = (a = ArrayNode(randn(Float32, 3, 4)), b = ArrayNode(randn(Float32, 3, 4))) |> ProductNode

    m1 = reflectinmodel(x1, layerbuilder; single_key_identity=false)
    m1_ski = reflectinmodel(x1, layerbuilder; single_key_identity=true)
    m1_ski_fsm = reflectinmodel(x1, layerbuilder; fsm=fsm, single_key_identity=true)
    m2 = reflectinmodel(x2, layerbuilder; single_key_identity=false)
    m2_ski = reflectinmodel(x2, layerbuilder; single_key_identity=true)
    m2_ski_fsm = reflectinmodel(x2, layerbuilder; fsm=fsm, single_key_identity=true)
    m3 = reflectinmodel(x3, layerbuilder; single_key_identity=false)
    m3_ski = reflectinmodel(x3, layerbuilder; single_key_identity=true)
    m3_ski_fsm = reflectinmodel(x3, layerbuilder; fsm=fsm, single_key_identity=true)

    for m in [m1, m1_ski, m1_ski_fsm]
        @test eltype(m(x1).data) == Float32
        @test size(m(x1).data) == (2, 4)
        @inferred m(x1)
        @test m isa ProductModel
        @test m.ms[1] isa ArrayModel
    end

    for m in [m2, m2_ski, m2_ski_fsm]
        @test eltype(m(x2).data) == Float32
        @test size(m(x2).data) == (2, 4)
        @inferred m(x2)
        @test m isa ProductModel
        @test m.ms[1] isa ArrayModel
    end

    for m in [m3, m3_ski, m3_ski_fsm]
        @test eltype(m(x3).data) == Float32
        @test size(m(x3).data) == (2, 4)
        @inferred m(x3)
        @test m isa ProductModel
        @test m.ms[1] isa ArrayModel
        @test m.ms[2] isa ArrayModel
    end

    @test m1.m isa ArrayModel{<:Dense}
    @test m2.m isa ArrayModel{<:Dense}
    @test m3.m isa ArrayModel{<:Dense}
    @test m1_ski.m isa IdentityModel
    @test m2_ski.m isa IdentityModel
    @test m3_ski.m isa ArrayModel{<:Dense}
    # fsm overrides ski
    @test m1_ski_fsm.m isa ArrayModel{<:Dense}
    @test m2_ski_fsm.m isa ArrayModel{<:Dense}
    @test m3_ski_fsm.m isa ArrayModel{<:Dense}
end

# array model for matrices with one row should implement identity
@testset "single scalar as identity" begin
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    x1 = ArrayNode(randn(Float32, 1, 3))
    x2 = BagNode(ArrayNode(randn(Float32, 1, 5)), [1:2, 3:5])
    x3 = (a = ArrayNode(randn(Float32, 1, 4)), b = ArrayNode(randn(Float32, 2, 4))) |> ProductNode

    m1 = reflectinmodel(x1, layerbuilder; single_scalar_identity=false)
    m1_sci = reflectinmodel(x1, layerbuilder; single_scalar_identity=true)
    m2 = reflectinmodel(x2, layerbuilder; single_scalar_identity=false)
    m2_sci = reflectinmodel(x2, layerbuilder; single_scalar_identity=true)
    m3 = reflectinmodel(x3, layerbuilder; single_scalar_identity=false)
    m3_sci = reflectinmodel(x3, layerbuilder; single_scalar_identity=true)

    @test size(m1(x1).data) == (2, 3)
    @test eltype(m1(x1).data) == Float32
    @test m1 isa ArrayModel
    @test size(m1_sci(x1).data) == (1, 3)
    @test eltype(m1_sci(x1).data) == Float32
    @test m1_sci isa ArrayModel
    @inferred m1(x1)

    @test size(m2(x2).data) == (2, 2)
    @test eltype(m2(x2).data) == Float32
    @test m2 isa BagModel
    @test m2.im isa ArrayModel{<:Dense}
    @test size(m2_sci(x2).data) == (2, 2)
    @test eltype(m2_sci(x2).data) == Float32
    @test m2_sci isa BagModel
    @test m2_sci.im isa IdentityModel
    @inferred m2(x2)

    @test size(m3(x3).data) == (2, 4)
    @test eltype(m3(x3).data) == Float32
    @test m3 isa ProductModel
    @test m3.ms[1] isa ArrayModel{<:Dense}
    @test m3.ms[2] isa ArrayModel{<:Dense}
    @test size(m3_sci(x3).data) == (2, 4)
    @test eltype(m3_sci(x3).data) == Float32
    @test m3_sci isa ProductModel
    @test m3_sci.ms[1] isa IdentityModel
    @test m3_sci.ms[2] isa ArrayModel{<:Dense}
    @inferred m3(x3)
end

@testset "model aggregation grad check w.r.t. inputs" begin
    for (bags1, bags2, bags3) in BAGS3
        layerbuilder(k) = Dense(k, 2, rand(ACTIVATIONS)) |> f64
        abuilder(d) = BagCount(all_aggregations(Float64, d))
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)

        for ds in [x -> ArrayNode(x),
                   x -> BagNode(ArrayNode(x), bags1)]
            m = reflectinmodel(ds(x), layerbuilder, abuilder)
            @inferred m(ds(x))
            @test gradtest(x -> m(ds(x)).data, x)
        end

        for ds in [(x, y) -> ProductNode((ArrayNode(x), ArrayNode(y))),
                   (x, y) -> ProductNode((a=BagNode(ArrayNode(x), bags1), b=BagNode(ArrayNode(y), bags2)))]
            m = reflectinmodel(ds(x, y), layerbuilder, abuilder)
            @inferred m(ds(x, y))
            @test gradtest((x, y) -> m(ds(x, y)).data, x, y)
        end

        ds = z -> BagNode(BagNode(ArrayNode(z), bags3), bags1)
        m = reflectinmodel(ds(z), layerbuilder, abuilder)
        @inferred m(ds(z))
        @test gradtest(z -> m(ds(z)).data, z)
    end
end

@testset "model aggregation grad check w.r.t. inputs weighted" begin
    for (bags1, bags2, bags3) in BAGS3
        layerbuilder(k) = Dense(k, 2, rand(ACTIVATIONS)) |> f64
        abuilder(d) = BagCount(all_aggregations(Float64, d))
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)
        w1 = abs.(randn(4)) .+ 0.1
        w2 = abs.(randn(4)) .+ 0.1
        w3 = abs.(randn(8)) .+ 0.1

        ds = x -> WeightedBagNode(ArrayNode(x), bags1, w1)
        m = reflectinmodel(ds(x), layerbuilder, abuilder)
        @inferred m(ds(x))
        @test gradtest(x -> m(ds(x)).data, x)

        ds = (x, y) -> ProductNode((WeightedBagNode(ArrayNode(x), bags1, w1),
                                    WeightedBagNode(ArrayNode(y), bags2, w2))) 
        m = reflectinmodel(ds(x, y), layerbuilder, abuilder)
        @inferred m(ds(x, y))
        @test gradtest((x, y) -> m(ds(x, y)).data, x, y)

        ds = z -> WeightedBagNode(WeightedBagNode(ArrayNode(z), bags3, w3), bags1, w1)
        m = reflectinmodel(ds(z), layerbuilder, abuilder)
        @inferred m(ds(z))
        @test gradtest(z -> m(ds(z)).data, z)
    end
end

@testset "model aggregation grad check w.r.t. params" begin
    for (bags1, bags2, bags3) in BAGS3
        layerbuilder(k) = Dense(k, 2, rand(ACTIVATIONS)) |> f64
        abuilder(d) = BagCount(all_aggregations(Float64, d))
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)

        for ds in [
                   ArrayNode(x),
                   BagNode(ArrayNode(x), bags1),
                   ProductNode((ArrayNode(x), ArrayNode(y))),
                   ProductNode((a=BagNode(ArrayNode(y), bags1), b=BagNode(ArrayNode(x), bags2))),
                   BagNode(BagNode(ArrayNode(z), bags3), bags1)
                  ]
            m = reflectinmodel(ds, layerbuilder, abuilder)
            @inferred m(ds)
            @test gradtest(() -> m(ds).data, Flux.params(m))
        end
    end
end

@testset "model aggregation grad check w.r.t. params weighted" begin
    for (bags1, bags2, bags3) in BAGS3
        layerbuilder(k) = Dense(k, 2) |> f64
        abuilder(d) = BagCount(all_aggregations(Float64, d))
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)
        w1 = abs.(randn(4)) .+ 0.1
        w2 = abs.(randn(4)) .+ 0.1
        w3 = abs.(randn(8)) .+ 0.1

        for ds in [
                   WeightedBagNode(ArrayNode(x), bags1, w1),
                   ProductNode((WeightedBagNode(ArrayNode(x), bags1, w1),
                                WeightedBagNode(ArrayNode(y), bags2, w2))) ,
                   WeightedBagNode(WeightedBagNode(ArrayNode(z), bags3, w3), bags1, w1)
                  ]
            m = reflectinmodel(ds, layerbuilder, abuilder)
            @inferred m(ds)
            @test gradtest(() -> m(ds).data, Flux.params(m))
        end
    end
end
