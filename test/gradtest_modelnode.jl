# in this file we use Float64 to compute precise gradients

# use only activations without "kinks" for numerical gradient checking
# see e.g. https://stackoverflow.com/questions/40623512/how-to-check-relu-gradient
const ACTIVATIONS = [identity, σ, swish, softplus, logcosh, mish, tanhshrink, lisht]
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

println("<HEARTBEAT>")

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
