# use only activations without "kinks" for numerical gradient checking
# see e.g. https://stackoverflow.com/questions/40623512/how-to-check-relu-gradient
const ACTIVATIONS = [identity, σ, swish, softplus, logcosh, mish, tanhshrink, lisht]
const LAYERBUILDER = k -> Flux.Dense(k, 2, rand(ACTIVATIONS))
const ABUILDER = d -> BagCount(all_aggregations(Float32, d))

@testset "constructor logic" begin
    d = Dense(2, 2)
    a = SegmentedMeanMax(2)
    m1 = ArrayModel(d)

    for m in [tuple(), tuple(identity), tuple(d)]
        m2 = BagModel(m1, a, m...)
        @test m2 isa BagModel{ArrayModel{typeof(d)}, typeof(a),
                isempty(m) ? typeof(identity) : typeof(only(m))}
        @test m2 == BagModel(d, a, m...)

        m3 = ProductModel(tuple(m2), m...)
        @test m3 isa ProductModel{Tuple{typeof(m2)}, isempty(m) ? typeof(identity) : typeof(only(m))}
        @test m3 == ProductModel(m2, m...)

        m4 = ProductModel((m1, m1), m...)
        @test m4 isa ProductModel{Tuple{typeof(m1), typeof(m1)},
                isempty(m) ? typeof(identity) : typeof(only(m))}
        @test m4 == ProductModel((d, d), m...) == ProductModel((m1, d), m...) ==
                    ProductModel((d, m1), m...)

        m4 = ProductModel((a=m1, b=m2), m...)
        @test m4 isa ProductModel{NamedTuple{(:a, :b), Tuple{typeof(m1), typeof(m2)}},
                isempty(m) ? typeof(identity) : typeof(only(m))}
        @test m4 == ProductModel((a=d, b=m2), m...) ==
                    ProductModel(a=m1, b=m2, m...) ==
                    ProductModel(a=d, b=m2, m...) == 
                    ProductModel(m...; a=m1, b=m2) ==
                    ProductModel(m...; a=d, b=m2)
    end

    m5 = LazyModel{:Test}(m1)
    @test m5 isa LazyModel{:Test, typeof(m1)}
    @test m5 == LazyModel{:Test}(d) ==
                LazyModel(:Test, m1) == LazyModel(:Test, d)
end

@testset "matrix model" begin
    x = ArrayNode(randn(Float32, 4, 5))
    m = reflectinmodel(x, LAYERBUILDER)
    @test m isa ArrayModel

    @test size(m(x)) == (2, 5)
    @test eltype(m(x)) == Float32

    @test m(x) == m.m(x.data)
    @inferred m(x)
end

@testset "bag model" begin
    x = BagNode(randn(Float32, 4, 4), [1:2, 3:4])
    m = reflectinmodel(x, LAYERBUILDER)
    @test m isa BagModel
    @test m.im isa ArrayModel

    @test size(m(x)) == (2, 2)
    @test eltype(m(x)) == Float32

    @test m(x) == m.bm(m.a(m.im.m(x.data.data), x.bags))
    @inferred m(x)
end

@testset "nested bag model" begin
    bn = BagNode(randn(Float32, 2, 8), [1:1, 2:2, 3:6, 7:8])
    x = BagNode(bn, [1:2, 3:4])

    m = reflectinmodel(x, LAYERBUILDER)
    @test m isa BagModel
    @test m.im isa BagModel
    @test m.im.im isa ArrayModel
    @test m.im.bm isa Dense
    @test m.bm isa Dense

    @test size(m(x)) == (2, 2)
    @test eltype(m(x)) == Float32

    @test m(x) == m.bm(m.a(m.im.bm(m.im.a(m.im.im.m(bn.data.data), bn.bags)), x.bags))
    @inferred m(x)

    a = BagNode(BagNode(randn(Float32, 2, 2),[1:2]),[1:1])
    b = BagNode(missing,[0:-1])
    c = BagNode(a.data[1:0], [0:-1])
    m = reflectinmodel(a)
    abc = catobs(a, b, c)
    bca = catobs(b, c, a)
    ma = m(a)
    mb = m(b)
    mc = m(c)
    mabc = m(abc)
    mbca = m(bca)
    @test mb ≈ mc
    @test mabc[:, 1] ≈ ma
    @test mabc[:, 2] ≈ mb
    @test mabc[:, 3] ≈ mc
    @test mbca[:, 1] ≈ mb
    @test mbca[:, 2] ≈ mc
    @test mbca[:, 3] ≈ ma

    @test m(b) == m.bm(m.a(b.data, b.bags))
    @test eltype(m(b)) == Float32
    @inferred m(b)
    for ds in [a, c, abc, bca]
        @test m(ds) == m.bm(m.a(m.im.bm(m.im.a(m.im.im.m(ds.data.data.data), ds.data.bags)), ds.bags))
        @test eltype(m(ds)) == Float32
        @inferred m(ds)
    end
end

@testset "product models" begin
    a = randn(Float32, 3, 4)
    b = randn(Float32, 4, 4)
    c = randn(Float32, 3, 4)
    x1 = ProductNode(; a, b)
    x2 = ProductNode(; b, a)
    x3 = ProductNode(; a, b, c)

    m = reflectinmodel(x1, LAYERBUILDER)
    @test m isa ProductModel
    @test m.ms[:a] isa ArrayModel
    @test m.ms[:b] isa ArrayModel

    @test m(x1) == m(x2) == m(x3)

    for x in [x1, x2, x3]
        @test m(x) == m.m(vcat(m.ms[:a].m(a),
                               m.ms[:b].m(b)))
        @test eltype(m(x)) == Float32
        @test size(m(x)) == (2, 4)
        @inferred m(x)
    end

    a = BagNode(randn(Float32, 3, 4), [1:2, 3:4])
    b = BagNode(randn(Float32, 4, 4), [1:1, 2:4])
    c = BagNode(randn(Float32, 2, 4), [1:1, 2:4])
    x1 = ProductNode((a, b))
    x2 = ProductNode((a, b, c))
    x3 = ProductNode(a)

    m = reflectinmodel(x1, LAYERBUILDER)
    @test m isa ProductModel
    @test m.ms[1] isa BagModel
    @test m.ms[1].im isa ArrayModel
    @test m.ms[1].bm isa Dense
    @test m.ms[2] isa BagModel
    @test m.ms[2].im isa ArrayModel
    @test m.ms[2].bm isa Dense

    ma = m.ms[1]
    mb = m.ms[2]
    for x in [x1, x2]
        @test m(x) == m.m(vcat(ma.bm(ma.a(ma.im.m(a.data.data), a.bags)),
                               mb.bm(mb.a(mb.im.m(b.data.data), b.bags))))
        @test size(m(x)) == (2, 2)
        @test eltype(m(x)) == Float32
        @inferred m(x)
    end
    @test_throws AssertionError m(x3)

    x = ProductNode([randn(Float32, 3, 4)])
    m = reflectinmodel(x, LAYERBUILDER)
    @test m isa ProductModel
    @test m.ms[1] isa ArrayModel

    @test size(m(x)) == (2, 4)
    @test eltype(m(x)) == Float32

    @test m(x) == m.m(m.ms[1].m(x.data[1].data))
    # ProductNodes with arrays are unstable because of vcat()
    # @inferred m(x)
end

@testset "product model keys and haskey" begin
    a = ArrayModel(Dense(2, 2))
    m1 = ProductModel([a, a])
    m2 = ProductModel((a, a))
    m3 = ProductModel(; a, b = a)
    @test keys(m1) == [1, 2]
    @test keys(m2) == [1, 2]
    @test keys(m3) == (:a, :b)

    @test haskey(m3, :a)
    @test haskey(m3, :b)
    @test !haskey(m3, :c)
end

@testset "reflectinmodel for missing + all_imputing" begin
    f1 = x -> reflectinmodel(x)
    f2 = x -> reflectinmodel(x, d -> Flux.Chain(Dense(d, 10), Dense(10, 10)))
    f3 = x -> reflectinmodel(x; all_imputing=true)
    _get_first(x::ArrayModel{<:Flux.Chain}) = x.m[1]
    _get_first(x) = x.m

    x1 = maybehot(2, 1:3) |> ArrayNode
    x2 = maybehot(missing, 1:3) |> ArrayNode
    for f in [f1, f2, f3], x in [x1, x2]
        @test _get_first(f(x)) isa PostImputingDense
        @inferred f(x)(x)
    end

    x1 = onehot(2, 1:3) |> ArrayNode
    for f in [f1, f2, f3]
        @test _get_first(f(x1)) isa Dense
        @inferred f(x1)(x1)
    end

    x1 = maybehotbatch([1, 2, 3], 1:3) |> ArrayNode
    x2 = maybehotbatch([1, 2, missing], 1:3) |> ArrayNode
    x3 = maybehotbatch(fill(missing, 3), 1:3) |> ArrayNode
    for f in [f1, f2, f3], x in [x1, x2, x3]
        @test _get_first(f(x)) isa PostImputingDense
        @inferred f(x)(x)
    end

    x1 = onehotbatch([1, 2, 3], 1:3) |> ArrayNode
    for f in [f1, f2, f3]
        @test _get_first(f(x1)) isa Dense
        @inferred f(x1)(x1)
    end

    x1 = NGramMatrix(["a", "b", "c"]) |> ArrayNode
    x2 = NGramMatrix(["a", missing, "c"]) |> ArrayNode
    x3 = NGramMatrix(fill(missing, 3)) |> ArrayNode
    for f in [f1, f2]
        @test _get_first(f(x1)) isa Flux.Dense{T, <:Matrix} where T
        @test _get_first(f(x2)) isa PostImputingDense
        @test _get_first(f(x3)) isa PostImputingDense
        for x in [x1, x2, x3] @inferred f(x)(x) end
    end
    for x in [x1, x2, x3]
        @test _get_first(f3(x)) isa PostImputingDense
        @inferred f3(x)(x)
    end

    x1 = rand([1, 2], 3, 3) |> ArrayNode
    x2 = rand([1, 2, missing], 3, 3) |> ArrayNode
    x3 = fill(missing, 3, 3) |> ArrayNode
    for f in [f1, f2]
        @test _get_first(f(x1)) isa Flux.Dense{T, <:Matrix} where T
        @test _get_first(f(x2)) isa PreImputingDense
        @test _get_first(f(x3)) isa PreImputingDense
        for x in [x1, x2, x3] @inferred f(x)(x) end
    end
    for x in [x1, x2, x3]
        @test _get_first(f3(x)) isa PreImputingDense
        @inferred f3(x)(x)
    end

    # BagNode and ProductNode submodels are not mapped
    n = BagNode(zeros(2, 2), [1:1, 2:2])
    @test f3(n).bm isa Flux.Dense{T, <:Matrix} where T
    n = ProductNode(a = zeros(2, 2), b = zeros(2, 2))
    @test f3(n).m isa Flux.Dense{T, <:Matrix} where T
end

# pn.m should be identity for any product node pn with a single key
@testset "single key dictionary reflect in model" begin
    fsm = Dict("" => LAYERBUILDER)

    x1 = (ArrayNode(randn(Float32, 3, 4)),) |> ProductNode
    x2 = (a = ArrayNode(randn(Float32, 3, 4)),) |> ProductNode
    x3 = (a = ArrayNode(randn(Float32, 3, 4)), b = ArrayNode(randn(Float32, 3, 4))) |> ProductNode

    m1 = reflectinmodel(x1, LAYERBUILDER; single_key_identity=false)
    m1_ski = reflectinmodel(x1, LAYERBUILDER; single_key_identity=true)
    m1_ski_fsm = reflectinmodel(x1, LAYERBUILDER; fsm=fsm, single_key_identity=true)
    m2 = reflectinmodel(x2, LAYERBUILDER; single_key_identity=false)
    m2_ski = reflectinmodel(x2, LAYERBUILDER; single_key_identity=true)
    m2_ski_fsm = reflectinmodel(x2, LAYERBUILDER; fsm=fsm, single_key_identity=true)
    m3 = reflectinmodel(x3, LAYERBUILDER; single_key_identity=false)
    m3_ski = reflectinmodel(x3, LAYERBUILDER; single_key_identity=true)
    m3_ski_fsm = reflectinmodel(x3, LAYERBUILDER; fsm=fsm, single_key_identity=true)

    for m in [m1, m1_ski, m1_ski_fsm]
        @test eltype(m(x1)) == Float32
        @test size(m(x1)) == (2, 4)
        @inferred m(x1)
        @test m isa ProductModel
        @test m.ms[1] isa ArrayModel
    end

    for m in [m2, m2_ski, m2_ski_fsm]
        @test eltype(m(x2)) == Float32
        @test size(m(x2)) == (2, 4)
        @inferred m(x2)
        @test m isa ProductModel
        @test m.ms[1] isa ArrayModel
    end

    for m in [m3, m3_ski, m3_ski_fsm]
        @test eltype(m(x3)) == Float32
        @test size(m(x3)) == (2, 4)
        @inferred m(x3)
        @test m isa ProductModel
        @test m.ms[1] isa ArrayModel
        @test m.ms[2] isa ArrayModel
    end

    @test m1.m isa Dense
    @test m2.m isa Dense
    @test m3.m isa Dense
    @test m1_ski.m ≡ identity
    @test m2_ski.m ≡ identity
    @test m3_ski.m isa Dense
    # fsm overrides ski
    @test m1_ski_fsm.m isa Dense
    @test m2_ski_fsm.m isa Dense
    @test m3_ski_fsm.m isa Dense
end

# array model for matrices with one row should implement identity
@testset "single scalar as identity" begin
    x1 = ArrayNode(randn(Float32, 1, 3))
    x2 = BagNode(randn(Float32, 1, 5), [1:2, 3:5])
    x3 = (a = ArrayNode(randn(Float32, 1, 4)), b = ArrayNode(randn(Float32, 2, 4))) |> ProductNode

    m1 = reflectinmodel(x1, LAYERBUILDER; single_scalar_identity=false)
    m1_sci = reflectinmodel(x1, LAYERBUILDER; single_scalar_identity=true)
    m2 = reflectinmodel(x2, LAYERBUILDER; single_scalar_identity=false)
    m2_sci = reflectinmodel(x2, LAYERBUILDER; single_scalar_identity=true)
    m3 = reflectinmodel(x3, LAYERBUILDER; single_scalar_identity=false)
    m3_sci = reflectinmodel(x3, LAYERBUILDER; single_scalar_identity=true)

    @test size(m1(x1)) == (2, 3)
    @test eltype(m1(x1)) == Float32
    @test m1 isa ArrayModel
    @test size(m1_sci(x1)) == (1, 3)
    @test eltype(m1_sci(x1)) == Float32
    @test m1_sci isa ArrayModel
    @inferred m1(x1)

    @test size(m2(x2)) == (2, 2)
    @test eltype(m2(x2)) == Float32
    @test m2 isa BagModel
    @test m2.im isa ArrayModel{<:Dense}
    @test size(m2_sci(x2)) == (2, 2)
    @test eltype(m2_sci(x2)) == Float32
    @test m2_sci isa BagModel
    @test m2_sci.im isa ArrayModel{typeof(identity)}
    @inferred m2(x2)

    @test size(m3(x3)) == (2, 4)
    @test eltype(m3(x3)) == Float32
    @test m3 isa ProductModel
    @test m3.ms[1] isa ArrayModel{<:Dense}
    @test m3.ms[2] isa ArrayModel{<:Dense}
    @test size(m3_sci(x3)) == (2, 4)
    @test eltype(m3_sci(x3)) == Float32
    @test m3_sci isa ProductModel
    @test m3_sci.ms[1] isa ArrayModel{typeof(identity)}
    @test m3_sci.ms[2] isa ArrayModel{<:Dense}
    @inferred m3(x3)
end

@testset "model aggregation grad check w.r.t. inputs" begin
    for (bags1, bags2, bags3) in BAGS3
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)

        for ds in [x -> ArrayNode(x),
                   x -> BagNode(x, bags1)]
            m = reflectinmodel(ds(x), LAYERBUILDER, ABUILDER) |> f64
            @inferred m(ds(x))
            @test @gradtest x -> m(ds(x))
        end

        for ds in [(x, y) -> ProductNode((x, y)),
                   (x, y) -> ProductNode(a=BagNode(x, bags1), b=BagNode(y, bags2))]
            m = reflectinmodel(ds(x, y), LAYERBUILDER, ABUILDER) |> f64
            @inferred m(ds(x, y))
            @test @gradtest (x, y) -> m(ds(x, y))
        end

        ds = z -> BagNode(BagNode(z, bags3), bags1)
        m = reflectinmodel(ds(z), LAYERBUILDER, ABUILDER) |> f64
        @inferred m(ds(z))
        @test @gradtest z -> m(ds(z))
    end
end

@testset "model aggregation grad check w.r.t. inputs weighted" begin
    for (bags1, bags2, bags3) in BAGS3
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)
        w1 = abs.(randn(4)) .+ 0.1
        w2 = abs.(randn(4)) .+ 0.1
        w3 = abs.(randn(8)) .+ 0.1

        ds = (x, w1) -> WeightedBagNode(x, bags1, w1)
        m = reflectinmodel(ds(x, w1), LAYERBUILDER, ABUILDER) |> f64
        @inferred m(ds(x, w1))
        @test @gradtest x -> m(ds(x, w1))

        ds = (x, y, w1, w2) -> ProductNode((WeightedBagNode(x, bags1, w1),
                                            WeightedBagNode(y, bags2, w2)))
        m = reflectinmodel(ds(x, y, w1, w2), LAYERBUILDER, ABUILDER) |> f64
        @inferred m(ds(x, y, w1, w2))
        @test @gradtest (x, y) -> m(ds(x, y, w1, w2))
        #
        ds = (z, w3, w1) -> WeightedBagNode(WeightedBagNode(z, bags3, w3), bags1, w1)
        m = reflectinmodel(ds(z, w3, w1), LAYERBUILDER, ABUILDER) |> f64
        @inferred m(ds(z, w3, w1))
        @test @gradtest z -> m(ds(z, w3, w1))
    end
end

@testset "model aggregation grad check w.r.t. params" begin
    for (bags1, bags2, bags3) in BAGS3
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)

        for ds in [
                   ArrayNode(x),
                   BagNode(x, bags1),
                   ProductNode((x, y)),
                   ProductNode(a=BagNode(y, bags1), b=BagNode(x, bags2)),
                   BagNode(BagNode(z, bags3), bags1)
                  ]
            m = reflectinmodel(ds, LAYERBUILDER, ABUILDER) |> f64
            @inferred m(ds)
            @test @pgradtest m -> m(ds)
        end
    end
end

@testset "model aggregation grad check w.r.t. params weighted" begin
    for (bags1, bags2, bags3) in BAGS3
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)
        w1 = abs.(randn(4)) .+ 0.1
        w2 = abs.(randn(4)) .+ 0.1
        w3 = abs.(randn(8)) .+ 0.1

        for ds in [
                   WeightedBagNode(x, bags1, w1),
                   ProductNode((WeightedBagNode(x, bags1, w1),
                                WeightedBagNode(y, bags2, w2))) ,
                   WeightedBagNode(WeightedBagNode(z, bags3, w3), bags1, w1)
                  ]
            m = reflectinmodel(ds, LAYERBUILDER, ABUILDER) |> f64
            @inferred m(ds)
            @test @pgradtest m -> m(ds)
        end
    end
end

@testset "simple named tuple model with reduce catobs in gradient" begin
    layerbuilder(k) = Dense(k, 2, relu)
    x = ProductNode(node1 = BagNode(randn(Float32, 3, 4), [1:2, 3:4]),
                    node2 = BagNode(randn(Float32, 4, 4), [1:1, 2:4]))
    m = reflectinmodel(x, layerbuilder)
    ps = Flux.params(m)
    vec_grad = gradient(() -> sum(m([x, x])), ps)
    @test vec_grad isa Grads
    reduced = reduce(catobs, [x, x])
    orig_grad = gradient(() -> sum(m(reduced)), ps)
    @test all(p -> vec_grad[p] == orig_grad[p], ps)
end

@testset "simple named tuple model with minibatching from Flux (MLUtils)" begin
    Random.seed!(0)
    layerbuilder(k) = Dense(k, 2, relu)
    x = ProductNode(node1 = BagNode(randn(Float32, 3, 6), [1:2, 3:4, 5:5, 6:6]),
                    node2 = BagNode(randn(Float32, 4, 6), [1:1, 2:4, 5:6, 0:-1]))
    m = reflectinmodel(x, layerbuilder)
    ps = Flux.params(m)
    mbs = Flux.DataLoader(x, batchsize=4, shuffle=true)
    mb = first(mbs)
    mb_grad = gradient(() -> sum(m(mb)), ps)
    @test mb_grad isa Grads
    reduced = reduce(catobs, [x[3], x[2], x[1], x[4]])
    orig_grad = gradient(() -> sum(m(mb)), ps)
    @test all(p -> mb_grad[p] == orig_grad[p], ps)
end
