using Flux, Test, Mill
using Mill: reflectinmodel
using Combinatorics

# use only activations without "kinks" for numerical gradient checking
# see e.g. https://stackoverflow.com/questions/40623512/how-to-check-relu-gradient
const ACTIVATIONS = [identity, σ, swish, softplus, logcosh, mish, tanhshrink, lisht]

@testset "aggregation grad check w.r.t. input" begin
    for bags in BAGS2
        d = rand(1:20)
        x = randn(d, 10)
        w = abs.(randn(size(x, 2))) .+ 0.1
        w_mat = abs.(randn(size(x))) .+ 0.1

        # generate all combinations of aggregations
        anames = ["Sum", "Mean", "Max", "PNorm", "LSE"]
        for idxs in powerset(collect(1:length(anames)))
            !isempty(idxs) || continue
            # not a thorough testing of all functions, but fast enough
            length(idxs) <= 3 || continue
            # for idxs in permutations(idxs)

            s = Symbol("Segmented", anames[idxs]...)
            a = @eval $s($d)
            @test mgradtest(x) do x
                a(x, bags)
            end
            @test mgradtest(x) do x
                a(x, bags, w)
            end
            @test mgradtest(x) do x
                a(x, bags, w_mat)
            end
        end
    end
end

@testset "aggregation grad check w.r.t. agg params" begin
    # r_map and p_map are stable
    @test first(gradient(softplus, 10000)) ≈ σ(10000) ≈ 1.0
    @test first(gradient(softplus, -10000)) ≈ σ(-10000) ≈ 0
     
    fs = [:SegmentedSum, :SegmentedMean, :SegmentedMax, :SegmentedPNorm, :SegmentedLSE]
    params = [(:C1,), (:C2,), (:C3,), (:ρ1, :c, :C4), (:ρ2, :C5)]

    for idxs in powerset(collect(1:length(fs)))
        !isempty(idxs) || continue;
        length(idxs) <= 2 || continue

        d = rand(1:20)
        x = randn(d, 0)
        as = []; cs = []; rs = []
        for (f, ps) in zip(fs[idxs], params[idxs])
            push!(rs, fill(:(randn($d)), length(ps))...)
            push!(as, ps...)
            push!(cs, Expr(:call, f, ps...))
        end
        @eval begin
            @test mgradtest($(map(eval, rs)...)) do $(as...)
                a = Aggregation($(cs...))
                a(missing, ScatteredBags([Int[], Int[]]))
            end
            @test mgradtest($(map(eval, rs)...)) do $(as...)
                a = Aggregation($(cs...))
                a(missing, AlignedBags([0:-1]), nothing)
            end
            @test mgradtest($(map(eval, rs)...)) do $(as...)
                a = Aggregation($(cs...))
                a($x, ScatteredBags([Int[]]))
            end
            @test mgradtest($(map(eval, rs)...)) do $(as...)
                a = Aggregation($(cs...))
                a($x, AlignedBags([0:-1, 0:-1]), nothing)
            end
        end

        for bags in BAGS2
            d = rand(1:20)
            x = randn(d, 10)
            w = abs.(randn(size(x, 2))) .+ 0.1
            w_mat = abs.(randn(size(x))) .+ 0.1
            as = []; cs = []; rs = []
            for (f, ps) in zip(fs[idxs], params[idxs])
                push!(rs, fill(:(randn($d)), length(ps))...)
                push!(as, ps...)
                push!(cs, Expr(:call, f, ps...))
            end
            @eval begin
                @test mgradtest($(map(eval, rs)...)) do $(as...)
                    a = Aggregation($(cs...))
                    a($x, $bags)
                end
                @test mgradtest($(map(eval, rs)...)) do $(as...)
                    a = Aggregation($(cs...))
                    a($x, $bags, $w)
                end
                @test mgradtest($(map(eval, rs)...)) do $(as...)
                    a = Aggregation($(cs...))
                    a($x, $bags, $w_mat)
                end
            end
        end
    end
end

@testset "derivative w.r.t weights in aggregations" begin
    for bags in BAGS2
        d = rand(1:20)
        x = randn(d, 10)
        w = abs.(randn(size(x, 2))) .+ 0.1
        w_mat = abs.(randn(size(x))) .+ 0.1

        a1 = f64(SegmentedSum(d))
        a2 = f64(SegmentedMean(d))
        a3 = f64(SegmentedMax(d))
        a4 = f64(SegmentedPNorm(d))
        a5 = f64(SegmentedLSE(d))
        for g in [
                  w -> a1(x, bags, w),
                  w -> a2(x, bags, w),
                  w -> a3(x, bags, w),
                  w -> a5(x, bags, w)
                 ]
            @test mgradtest(g, w)
        end
        for g in [
                  w_mat -> a1(x, bags, w_mat),
                  w_mat -> a2(x, bags, w_mat),
                  w_mat -> a3(x, bags, w_mat),
                  w_mat -> a5(x, bags, w_mat)
                 ]
            @test mgradtest(g, w_mat)
        end
        # for g in [
        #           w -> a4(x, bags, w_mat)
        #          ]
        #     # NOT IMPLEMENTED YET
        #     @test_throws Exception mgradtest(g, w_mat)
        # end
        # for g in [
        #           w -> a4(x, bags, w_mat)
        #          ]
        #     # NOT IMPLEMENTED YET
        #     @test_throws Exception mgradtest(g, w_mat)
        # end
    end
end

@testset "model aggregation grad check w.r.t. inputs" begin
    for (bags1, bags2, bags3) in BAGS3
        layerbuilder(k) = Dense(k, 2, rand(ACTIVATIONS))
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)

        n = ArrayNode(x)
        m = f64(reflectinmodel(n, layerbuilder))
        @test mgradtest(x) do x
            n = ArrayNode(x)
            m(n).data
        end

        bn = BagNode(ArrayNode(x), bags1)
        abuilder = d -> SegmentedPNormLSE(d)
        m = f64(reflectinmodel(bn, layerbuilder))
        @test mgradtest(x) do x
            bn = BagNode(ArrayNode(x), bags1)
            m(bn).data
        end

        tn = ProductNode((ArrayNode(x), ArrayNode(y)))
        m = f64(reflectinmodel(tn, layerbuilder))
        @test mgradtest(x, y) do x, y
            tn = ProductNode((ArrayNode(x), ArrayNode(y)))
            m(tn).data
        end

        tn = ProductNode((BagNode(ArrayNode(y), bags1), BagNode(ArrayNode(x), bags2)))
        abuilder = d -> SegmentedMeanMax(d)
        m = f64(reflectinmodel(tn, layerbuilder, abuilder))
        @test mgradtest(x, y) do x, y
            tn = ProductNode((BagNode(ArrayNode(y), bags1), BagNode(ArrayNode(x), bags2)))
            m(tn).data
        end

        bn = BagNode(ArrayNode(z), bags3)
        bnn = BagNode(bn, bags1)
        abuilder = d -> SegmentedSumMaxPNormLSE(d)
        m = f64(reflectinmodel(bnn, layerbuilder, abuilder))
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
        m = f64(reflectinmodel(bn, layerbuilder))
        @test mgradtest(x) do x
            bn = BagNode(ArrayNode(x), bags1, w)
            m(bn).data
        end

        tn = ProductNode((BagNode(ArrayNode(y), bags1, w), BagNode(ArrayNode(x), bags2, w2)))
        abuilder = d -> SegmentedMeanMax(d)
        m = f64(reflectinmodel(tn, layerbuilder, abuilder))
        @test mgradtest(x, y) do x, y
            tn = ProductNode((BagNode(ArrayNode(y), bags1, w), BagNode(ArrayNode(x), bags2, w2)))
            m(tn).data
        end

        bn = BagNode(ArrayNode(z), bags3, w3)
        bnn = BagNode(bn, bags1)
        abuilder = d -> SegmentedSumMaxPNormLSE(d)
        m = f64(reflectinmodel(bnn, layerbuilder, abuilder))
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
        m = f64(reflectinmodel(n, layerbuilder))
        a = rand(ACTIVATIONS)
        @test mgradtest(params(m)...) do W, b
            m = ArrayModel(Dense(W, b, a))
            m(n).data
        end

        bn = BagNode(ArrayNode(x), bags1)
        abuilder = d -> SegmentedPNormLSE(d)
        m = f64(reflectinmodel(bn, layerbuilder, abuilder))
        a1, a2 = rand(ACTIVATIONS, 2)
        @test mgradtest(params(m)...) do W1, b1, ρ1, c, C1, ρ2, C2, W2, b2
            m = BagModel(Dense(W1, b1, a1),
                         Aggregation(
                                     SegmentedPNorm(ρ1, c, C1),
                                     SegmentedLSE(ρ2, C2)
                                    ),
                         Dense(W2, b2, a2))
            m(bn).data
        end

        tn = ProductNode((ArrayNode(x), ArrayNode(y)))
        m = f64(reflectinmodel(tn, layerbuilder))
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
        m = f64(reflectinmodel(tn, layerbuilder, abuilder))
        a1, a2, a3, a4, a5 = rand(ACTIVATIONS, 5)
        @test mgradtest(params(m)...) do W1, b1, C11, C12, ρ11, c1, C13, ρ12, C14,
            W2, b2, W3, b3, C21, C22, ρ21, c2, C23, ρ22, C24, W4, b4, W5, b5
            m = ProductModel((
                              BagModel(
                                       Dense(W1, b1, a1),
                                       Aggregation(
                                                   SegmentedSum(C11),
                                                   SegmentedMax(C12),
                                                   SegmentedPNorm(ρ11, c1, C13),
                                                   SegmentedLSE(ρ12, C14)
                                                  ),
                                       Dense(W2, b2, a2)
                                      ),
                              BagModel(
                                       Dense(W3, b3, a3),
                                       Aggregation(
                                                   SegmentedSum(C21),
                                                   SegmentedMax(C22),
                                                   SegmentedPNorm(ρ21, c2, C23),
                                                   SegmentedLSE(ρ22, C24)
                                                  ),
                                       Dense(W4, b4, a4)
                                      ),
                             ), Dense(W5, b5, a5)) 
            m(tn).data
        end

        bn = BagNode(ArrayNode(z), bags3)
        bnn = BagNode(bn, bags1)
        abuilder = d -> SegmentedMeanMax(d)
        m = f64(reflectinmodel(bnn, layerbuilder, abuilder))
        a1, a2, a3 = rand(ACTIVATIONS, 3)
        @test mgradtest(params(m)...) do W1, b1, C11, C12, W2, b2, C21, C22, W3, b3
            m = BagModel(
                         BagModel(
                                  Dense(W1, b1, a1),
                                  Aggregation(
                                              SegmentedMean(C11),
                                              SegmentedMax(C12)
                                             ),
                                  Dense(W2, b2, a2)
                                 ),
                         Aggregation(
                                     SegmentedMean(C21),
                                     SegmentedMax(C22)
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
        m = f64(reflectinmodel(bn, layerbuilder, abuilder))
        a1, a2 = rand(ACTIVATIONS, 2)
        @test mgradtest(params(m)...) do W1, b1, ρ1, c, C1, ρ2, C2, W2, b2
            m = BagModel(Dense(W1, b1, a1),
                         Aggregation(
                                     SegmentedPNorm(ρ1, c, C1),
                                     SegmentedLSE(ρ2, C2)
                                    ),
                         Dense(W2, b2, a2))
            m(bn).data
        end

        tn = ProductNode((BagNode(ArrayNode(y), bags1, w), BagNode(ArrayNode(x), bags2, w2)))
        abuilder = d -> SegmentedSumMaxPNormLSE(d)
        m = f64(reflectinmodel(tn, layerbuilder, abuilder))
        a1, a2, a3, a4, a5 = rand(ACTIVATIONS, 5)
        @test mgradtest(params(m)...) do W1, b1, C11, C12, ρ11, c1, C13, ρ12, C14,
            W2, b2, W3, b3, C21, C22, ρ21, c2, C23, ρ22, C24, W4, b4, W5, b5
            m = ProductModel((
                              BagModel(
                                       Dense(W1, b1, a1),
                                       Aggregation(
                                                   SegmentedSum(C11),
                                                   SegmentedMax(C12),
                                                   SegmentedPNorm(ρ11, c1, C13),
                                                   SegmentedLSE(ρ12, C14),
                                                  ),
                                       Dense(W2, b2, a2)
                                      ),
                              BagModel(
                                       Dense(W3, b3, a3),
                                       Aggregation(
                                                   SegmentedSum(C21),
                                                   SegmentedMax(C22),
                                                   SegmentedPNorm(ρ21, c2, C23),
                                                   SegmentedLSE(ρ22, C24)
                                                  ),
                                       Dense(W4, b4, a4)
                                      ),
                             ), Dense(W5, b5, a5)) 
            m(tn).data
        end

        bn = BagNode(ArrayNode(z), bags3, w3)
        bnn = BagNode(bn, bags1, w)
        abuilder = d -> SegmentedMeanMax(d)
        m = f64(reflectinmodel(bnn, layerbuilder, abuilder))
        a1, a2, a3 = rand(ACTIVATIONS, 3)
        @test mgradtest(params(m)...) do W1, b1, C11, C12, W2, b2, C21, C22, W3, b3
            m = BagModel(
                         BagModel(
                                  Dense(W1, b1, a1),
                                  Aggregation(
                                              SegmentedMean(C11),
                                              SegmentedMax(C12)
                                             ),
                                  Dense(W2, b2, a2)
                                 ),
                         Aggregation(
                                     SegmentedMean(C21),
                                     SegmentedMax(C22)
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
        m = f64(ProductModel((
                              a = ArrayModel(Dense(2, 2, a1)),
                              b = ArrayModel(Dense(3, 1, a2))
                             ), Dense(3, 2, a3)))
        @test mgradtest(params(m)...) do W1, b1, W2, b2, W3, b3 
            m = ProductModel((
                              a = ArrayModel(Dense(W1, b1, a1)),
            b = ArrayModel(Dense(W2, b2, a2))
           ), Dense(W3, b3, a3))
            sum(m(x).data)
        end
    end
end
