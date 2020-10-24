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
end

@testset "testing simple aggregation model" begin
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    x = BagNode(ArrayNode(randn(Float32, 4, 4)), [1:2, 3:4])
    m = reflectinmodel(x, layerbuilder)
    @test size(m(x).data) == (2, 2)
    @test m isa BagModel
    @test eltype(m(x).data) == Float32
end

@testset "testing simple tuple models" begin
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    x = ProductNode((ArrayNode(randn(Float32, 3, 4)), ArrayNode(randn(Float32, 4, 4))))
    m = reflectinmodel(x, layerbuilder)
    @test eltype(m(x).data) == Float32
    @test size(m(x).data) == (2, 4)
    @test m isa ProductModel
    @test m.ms[1] isa ArrayModel
    @test m.ms[2] isa ArrayModel

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

    a = BagNode(BagNode(ArrayNode(randn(2,2)),[1:2]),[1:1])
    b = BagNode(missing,[0:-1])
    c = BagNode(a.data[1:0], [0:-1])
    m = reflectinmodel(a, d -> Dense(d,2), d -> SegmentedMeanMax(d))
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
end

# pn.m should be identity for any product node pn with a single key
@testset "single key dictionary reflect in model" begin
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    x1 = (ArrayNode(randn(Float32, 3, 4)),) |> ProductNode
    x2 = (a = ArrayNode(randn(Float32, 3, 4)),) |> ProductNode
    x3 = (a = ArrayNode(randn(Float32, 3, 4)), b = ArrayNode(randn(Float32, 3, 4))) |> ProductNode

    m1 = reflectinmodel(x1, layerbuilder; single_key_identity=false)
    m1_ski = reflectinmodel(x1, layerbuilder; single_key_identity=true)
    m2 = reflectinmodel(x2, layerbuilder; single_key_identity=false)
    m2_ski = reflectinmodel(x2, layerbuilder; single_key_identity=true)
    m3 = reflectinmodel(x3, layerbuilder; single_key_identity=false)
    m3_ski = reflectinmodel(x3, layerbuilder; single_key_identity=true)

    for m in [m1, m1_ski]
        @test eltype(m(x1).data) == Float32
        @test size(m(x1).data) == (2, 4)
        @test m isa ProductModel
        @test m.ms[1] isa ArrayModel
    end

    for m in [m2, m2_ski]
        @test eltype(m(x2).data) == Float32
        @test size(m(x2).data) == (2, 4)
        @test m isa ProductModel
        @test m.ms[1] isa ArrayModel
    end

    for m in [m3, m3_ski]
        @test eltype(m(x3).data) == Float32
        @test size(m(x3).data) == (2, 4)
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

    @test size(m2(x2).data) == (2, 2)
    @test eltype(m2(x2).data) == Float32
    @test m2 isa BagModel
    @test m2.im isa ArrayModel{<:Dense}
    @test size(m2_sci(x2).data) == (2, 2)
    @test eltype(m2_sci(x2).data) == Float32
    @test m2_sci isa BagModel
    @test m2_sci.im isa IdentityModel

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
end

# Defining this is a bad idea - in Flux all models do not implement == and hash
# it may break AD
# @testset "testing equals and hash" begin
#     # TODO lazy and missing models should be tested in a similar fashion
#     @eval layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
#     @eval x1 = ArrayNode(randn(Float32, 3, 4)) 
#     @eval x2 = ArrayNode(randn(Float32, 4, 4)) 
#     @eval x3 = BagNode(x1, [1:2, 3:4])
#     @eval x4 = BagNode(x2, [1:1, 2:4])
#     @eval x5 = ProductNode((x3, x4))

#     var(i, pref, suf="") = Symbol(pref * string(i) * suf)

#     for i in 1:5
#         @eval $(var(i, "m")) = reflectinmodel($(var(i, "x")), layerbuilder)
#         @eval $(var(i, "m", "c")) = deepcopy($(var(i, "m")))
#     end

#     for i in 1:5, j in 1:5
#         @show i,j
#         res1 = i == j
#         res2 = @eval $(var(i, "m")) == $(var(j, "m", "c"))
#         @test res1 == res2
#     end

#     for i in 1:5, j in 1:5
#         res1 = i == j
#         res2 = @eval hash($(var(i, "m"))) == hash($(var(j, "m", "c")))
#         @test res1 == res2
#     end
# end

# we use Float64 to compute precise gradients

@testset "model aggregation grad check w.r.t. inputs" begin
    for (bags1, bags2, bags3) in BAGS3
        layerbuilder(k) = Dense(k, 2, rand(ACTIVATIONS))
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)

        n = ArrayNode(x)
        m = reflectinmodel(n, layerbuilder) |> f64
        @test mgradtest(x) do x
            n = ArrayNode(x)
            m(n).data
        end

        bn = BagNode(ArrayNode(x), bags1)
        abuilder = d -> SegmentedPNormLSE(d)
        m = reflectinmodel(bn, layerbuilder) |> f64
        @test mgradtest(x) do x
            bn = BagNode(ArrayNode(x), bags1)
            m(bn).data
        end

        tn = ProductNode((ArrayNode(x), ArrayNode(y)))
        m = reflectinmodel(tn, layerbuilder) |> f64
        @test mgradtest(x, y) do x, y
            tn = ProductNode((ArrayNode(x), ArrayNode(y)))
            m(tn).data
        end

        tn = ProductNode((BagNode(ArrayNode(y), bags1), BagNode(ArrayNode(x), bags2)))
        abuilder = d -> SegmentedMeanMax(d)
        m = reflectinmodel(tn, layerbuilder, abuilder) |> f64
        @test mgradtest(x, y) do x, y
            tn = ProductNode((BagNode(ArrayNode(y), bags1), BagNode(ArrayNode(x), bags2)))
            m(tn).data
        end

        bn = BagNode(ArrayNode(z), bags3)
        bnn = BagNode(bn, bags1)
        abuilder = d -> SegmentedSumMaxPNormLSE(d)
        m = reflectinmodel(bnn, layerbuilder, abuilder) |> f64
        @test mgradtest(z) do z
            bn = BagNode(ArrayNode(z), bags3)
            bnn = BagNode(bn, bags1)
            m(bnn).data
        end
    end
end

@testset "model aggregation grad check w.r.t. inputs weighted" begin
    for (bags1, bags2, bags3) in BAGS3
        layerbuilder(k) = Dense(k, 2, rand(ACTIVATIONS))
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)
        w = abs.(randn(4)) .+ 0.1
        w2 = abs.(randn(4)) .+ 0.1
        w3 = abs.(randn(8)) .+ 0.1

        bn = BagNode(ArrayNode(x), bags1, w)
        abuilder = d -> SegmentedPNormLSE(d)
        m = reflectinmodel(bn, layerbuilder) |> f64
        @test mgradtest(x) do x
            bn = BagNode(ArrayNode(x), bags1, w)
            m(bn).data
        end

        tn = ProductNode((BagNode(ArrayNode(y), bags1, w), BagNode(ArrayNode(x), bags2, w2)))
        abuilder = d -> SegmentedMeanMax(d)
        m = reflectinmodel(tn, layerbuilder, abuilder) |> f64
        @test mgradtest(x, y) do x, y
            tn = ProductNode((BagNode(ArrayNode(y), bags1, w), BagNode(ArrayNode(x), bags2, w2)))
            m(tn).data
        end

        bn = BagNode(ArrayNode(z), bags3, w3)
        bnn = BagNode(bn, bags1)
        abuilder = d -> SegmentedSumMaxPNormLSE(d)
        m = reflectinmodel(bnn, layerbuilder, abuilder) |> f64
        @test mgradtest(z) do z
            bn = BagNode(ArrayNode(z), bags3, w3)
            bnn = BagNode(bn, bags1, w)
            m(bnn).data
        end
    end
end

@testset "model aggregation grad check w.r.t. params" begin
    for (bags1, bags2, bags3) in BAGS3
        layerbuilder(k) = Dense(k, 2)
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)

        n = ArrayNode(x)
        m = reflectinmodel(n, layerbuilder) |> f64
        a = rand(ACTIVATIONS)
        @test mgradtest(params(m)...) do W, b
            m = ArrayModel(Dense(W, b, a))
            m(n).data
        end

        bn = BagNode(ArrayNode(x), bags1)
        abuilder = d -> SegmentedPNormLSE(d)
        m = reflectinmodel(bn, layerbuilder, abuilder) |> f64
        a1, a2 = rand(ACTIVATIONS, 2)
        @test mgradtest(params(m)...) do W1, b1, ρ1, c, ψ1, ρ2, ψ2, W2, b2
            m = BagModel(Dense(W1, b1, a1),
                         Aggregation(
                                     SegmentedPNorm(ρ1, c, ψ1),
                                     SegmentedLSE(ρ2, ψ2)
                                    ),
                         Dense(W2, b2, a2))
            m(bn).data
        end

        tn = ProductNode((ArrayNode(x), ArrayNode(y)))
        m = reflectinmodel(tn, layerbuilder) |> f64
        a1, a2, a3 = rand(ACTIVATIONS, 3)
        @test mgradtest(params(m)...) do W1, b1, W2, b2, W3, b3
            m = ProductModel(ArrayModel.((
                                          Dense(W1, b1, a1),
                                          Dense(W2, b2, a2)
                                         )), Dense(W3, b3, a3)) 
            m(tn).data
        end

        tn = ProductNode((BagNode(ArrayNode(y), bags1), BagNode(ArrayNode(x), bags2)))
        abuilder = d -> SegmentedSumMaxPNormLSE(d)
        m = reflectinmodel(tn, layerbuilder, abuilder) |> f64
        a1, a2, a3, a4, a5 = rand(ACTIVATIONS, 5)
        @test mgradtest(params(m)...) do W1, b1, ψ11, ψ12, ρ11, c1, ψ13, ρ12, ψ14,
            W2, b2, W3, b3, ψ21, ψ22, ρ21, c2, ψ23, ρ22, ψ24, W4, b4, W5, b5
            m = ProductModel((
                              BagModel(
                                       Dense(W1, b1, a1),
                                       Aggregation(
                                                   SegmentedSum(ψ11),
                                                   SegmentedMax(ψ12),
                                                   SegmentedPNorm(ρ11, c1, ψ13),
                                                   SegmentedLSE(ρ12, ψ14)
                                                  ),
                                       Dense(W2, b2, a2)
                                      ),
                              BagModel(
                                       Dense(W3, b3, a3),
                                       Aggregation(
                                                   SegmentedSum(ψ21),
                                                   SegmentedMax(ψ22),
                                                   SegmentedPNorm(ρ21, c2, ψ23),
                                                   SegmentedLSE(ρ22, ψ24)
                                                  ),
                                       Dense(W4, b4, a4)
                                      ),
                             ), Dense(W5, b5, a5)) 
            m(tn).data
        end

        bn = BagNode(ArrayNode(z), bags3)
        bnn = BagNode(bn, bags1)
        abuilder = d -> SegmentedMeanMax(d)
        m = reflectinmodel(bnn, layerbuilder, abuilder) |> f64
        a1, a2, a3 = rand(ACTIVATIONS, 3)
        @test mgradtest(params(m)...) do W1, b1, ψ11, ψ12, W2, b2, ψ21, ψ22, W3, b3
            m = BagModel(
                         BagModel(
                                  Dense(W1, b1, a1),
                                  Aggregation(
                                              SegmentedMean(ψ11),
                                              SegmentedMax(ψ12)
                                             ),
                                  Dense(W2, b2, a2)
                                 ),
                         Aggregation(
                                     SegmentedMean(ψ21),
                                     SegmentedMax(ψ22)
                                    ),
                         Dense(W3, b3, a3)
                        )
            m(bnn).data
        end
    end
end

@testset "model aggregation grad check w.r.t. params weighted" begin
    for (bags1, bags2, bags3) in BAGS3
        layerbuilder(k) = Dense(k, 2)
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)
        w = abs.(randn(4)) .+ 0.1
        w2 = abs.(randn(4)) .+ 0.1
        w3 = abs.(randn(8)) .+ 0.1

        bn = BagNode(ArrayNode(x), bags1, w)
        abuilder = d -> SegmentedPNormLSE(d)
        m = reflectinmodel(bn, layerbuilder, abuilder) |> f64
        a1, a2 = rand(ACTIVATIONS, 2)
        @test mgradtest(params(m)...) do W1, b1, ρ1, c, ψ1, ρ2, ψ2, W2, b2
            m = BagModel(Dense(W1, b1, a1),
                         Aggregation(
                                     SegmentedPNorm(ρ1, c, ψ1),
                                     SegmentedLSE(ρ2, ψ2)
                                    ),
                         Dense(W2, b2, a2))
            m(bn).data
        end

        tn = ProductNode((BagNode(ArrayNode(y), bags1, w), BagNode(ArrayNode(x), bags2, w2)))
        abuilder = d -> SegmentedSumMaxPNormLSE(d)
        m = reflectinmodel(tn, layerbuilder, abuilder) |> f64
        a1, a2, a3, a4, a5 = rand(ACTIVATIONS, 5)
        @test mgradtest(params(m)...) do W1, b1, ψ11, ψ12, ρ11, c1, ψ13, ρ12, ψ14,
            W2, b2, W3, b3, ψ21, ψ22, ρ21, c2, ψ23, ρ22, ψ24, W4, b4, W5, b5
            m = ProductModel((
                              BagModel(
                                       Dense(W1, b1, a1),
                                       Aggregation(
                                                   SegmentedSum(ψ11),
                                                   SegmentedMax(ψ12),
                                                   SegmentedPNorm(ρ11, c1, ψ13),
                                                   SegmentedLSE(ρ12, ψ14),
                                                  ),
                                       Dense(W2, b2, a2)
                                      ),
                              BagModel(
                                       Dense(W3, b3, a3),
                                       Aggregation(
                                                   SegmentedSum(ψ21),
                                                   SegmentedMax(ψ22),
                                                   SegmentedPNorm(ρ21, c2, ψ23),
                                                   SegmentedLSE(ρ22, ψ24)
                                                  ),
                                       Dense(W4, b4, a4)
                                      ),
                             ), Dense(W5, b5, a5)) 
            m(tn).data
        end

        bn = BagNode(ArrayNode(z), bags3, w3)
        bnn = BagNode(bn, bags1, w)
        abuilder = d -> SegmentedMeanMax(d)
        m = reflectinmodel(bnn, layerbuilder, abuilder) |> f64
        a1, a2, a3 = rand(ACTIVATIONS, 3)
        @test mgradtest(params(m)...) do W1, b1, ψ11, ψ12, W2, b2, ψ21, ψ22, W3, b3
            m = BagModel(
                         BagModel(
                                  Dense(W1, b1, a1),
                                  Aggregation(
                                              SegmentedMean(ψ11),
                                              SegmentedMax(ψ12)
                                             ),
                                  Dense(W2, b2, a2)
                                 ),
                         Aggregation(
                                     SegmentedMean(ψ21),
                                     SegmentedMax(ψ22)
                                    ),
                         Dense(W3, b3, a3)
                        )
            m(bnn).data
        end
    end

    @testset "A gradient of ProductNode with a NamedTuple " begin
        a1, a2, a3 = rand(ACTIVATIONS, 3)
        x = ProductNode((
                         a = ArrayNode(randn(2, 4)),
        b = ArrayNode(randn(3, 4))
       ))
        m = ProductModel((
                          a = ArrayModel(Dense(2, 2, a1)),
                          b = ArrayModel(Dense(3, 1, a2))
                         ), Dense(3, 2, a3)) |> f64
        @test mgradtest(params(m)...) do W1, b1, W2, b2, W3, b3 
            m = ProductModel((
                              a = ArrayModel(Dense(W1, b1, a1)),
            b = ArrayModel(Dense(W2, b2, a2))
           ), Dense(W3, b3, a3))
            sum(m(x).data)
        end
    end
end
