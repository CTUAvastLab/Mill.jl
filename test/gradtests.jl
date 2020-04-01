using Flux, Test, Mill
using Mill: reflectinmodel, length2bags
using Combinatorics

BAGS = [
        length2bags([1 for _ in 1:10]),
        length2bags([2 for _ in 1:5]),
        length2bags([5, 5]),
        length2bags([10]),
length2bags([3, 4, 3]),
AlignedBags([1:3, 0:-1, 0:-1, 4:7, 0:-1, 8:10]),
AlignedBags([0:-1, 1:5, 0:-1, 0:-1, 0:-1, 6:10]),
ScatteredBags([collect(1:3), collect(7:10), collect(4:6)]),
ScatteredBags([collect(7:10), [], collect(1:3), [], collect(4:6), []]),
ScatteredBags([[], collect(1:10), []]),
]

BAGS2 = [
         (AlignedBags([1:2, 3:4, 0:-1]), ScatteredBags([[2,3,4], [1], []]), AlignedBags([1:4, 0:-1, 5:8, 0:-1])),
         (AlignedBags([0:-1, 1:2, 3:4]), ScatteredBags([[1], [2], [3, 4]]), AlignedBags([0:-1, 1:7, 0:-1, 8:8])),
         (AlignedBags([0:-1, 0:-1, 1:2, 3:4]), ScatteredBags([[2,4], [], [3, 1], []]), AlignedBags([1:1, 2:2, 0:-1, 3:8])),
         (AlignedBags([0:-1, 1:2, 3:4, 0:-1]), ScatteredBags([[], [1,3], [2,4], []]), AlignedBags([0:-1, 1:2, 3:6, 7:8]))
        ]

@testset "aggregation grad check w.r.t. input" begin
    for bags in BAGS
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
    for bags in BAGS
        d = rand(1:20)
        x = randn(d, 10)
        w = abs.(randn(size(x, 2))) .+ 0.1
        w_mat = abs.(randn(size(x))) .+ 0.1

        fs = [:SegmentedSum, :SegmentedMax, :SegmentedMean, :SegmentedPNorm, :SegmentedLSE]
        params = (:C1,), (:C2,), (:C3,), (:ρ1, :c, :C4), (:ρ2, :C5)

        for idxs in powerset(collect(1:length(fs)))
            !isempty(idxs) || continue;
            length(idxs) <= 2 || continue
            rs = []; as = []; cs = []; 
            for (f, ps) in zip(fs[idxs], params[idxs])
                push!(rs, fill(:(randn($d)), length(ps))...)
                push!(as, ps...)
                push!(cs, Expr(:call, f, ps...))
            end
            @eval begin
                @test mgradtest($(map(eval, rs)...)) do $(as...)
                    a = Aggregation($(cs...))
                    a(missing, $bags)
                end
                @test mgradtest($(map(eval, rs)...)) do $(as...)
                    a = Aggregation($(cs...))
                    a($x, $bags)
                end
                @test mgradtest($(map(eval, rs)...)) do $(as...)
                    a = Aggregation($(cs...))
                    a(missing, $bags, nothing)
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
    for bags in BAGS
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
    for (bags1, bags2, bags3) in BAGS2
        layerbuilder(k) = Dense(k, 2, NNlib.relu)
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
        abuilder = d -> SegmentedPNormLSESumMax(d)
        m = f64(reflectinmodel(bnn, layerbuilder, abuilder))
        @test mgradtest(z) do z
            bn = BagNode(ArrayNode(z), bags3)
            bnn = BagNode(bn, bags1)
            m(bnn).data
        end
    end
end

@testset "model aggregation grad check w.r.t. inputs weighted" begin
    for (bags1, bags2, bags3) in BAGS2
        layerbuilder(k) = Dense(k, 2, NNlib.relu)
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
        abuilder = d -> SegmentedPNormLSESumMax(d)
        m = f64(reflectinmodel(bnn, layerbuilder, abuilder))
        @test mgradtest(z) do z
            bn = BagNode(ArrayNode(z), bags3, w3)
            bnn = BagNode(bn, bags1, w)
            m(bnn).data
        end
    end
end

@testset "model aggregation grad check w.r.t. params" begin
    for (bags1, bags2, bags3) in BAGS2
        layerbuilder(k) = Dense(k, 2, NNlib.relu)
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)

        n = ArrayNode(x)
        m = f64(reflectinmodel(n, layerbuilder))
        @test mgradtest(params(m)...) do W, b
            m = ArrayModel(Dense(W, b, relu))
            m(n).data
        end

        bn = BagNode(ArrayNode(x), bags1)
        abuilder = d -> SegmentedPNormLSE(d)
        m = f64(reflectinmodel(bn, layerbuilder, abuilder))
        @test mgradtest(params(m)...) do W1, b1, ρ1, c, C1, ρ2, C2, W2, b2
            m = BagModel(Dense(W1, b1, relu),
                         Aggregation(
                                     SegmentedPNorm(ρ1, c, C1),
                                     SegmentedLSE(ρ2, C2)
                                    ),
                         Dense(W2, b2, σ))
            m(bn).data
        end

        tn = ProductNode((ArrayNode(x), ArrayNode(y)))
        m = f64(reflectinmodel(tn, layerbuilder))
        @test mgradtest(params(m)...) do W1, b1, W2, b2, W3, b3
            m = ProductModel(ArrayModel.((
                                          Dense(W1, b1, σ),
                                          Dense(W2, b2, relu)
                                         )), Dense(W3, b3, σ)) 
            m(tn).data
        end

        tn = ProductNode((BagNode(ArrayNode(y), bags1), BagNode(ArrayNode(x), bags2)))
        abuilder = d -> SegmentedPNormLSESumMax(d)
        m = f64(reflectinmodel(tn, layerbuilder, abuilder))
        @test mgradtest(params(m)...) do W1, b1, ρ11, c1, C11, ρ12, C12, C13, C14,
            W2, b2, W3, b3, ρ21, c2, C21, ρ22, C22, C23, C24, W4, b4, W5, b5
            m = ProductModel((
                              BagModel(
                                       Dense(W1, b1, σ),
                                       Aggregation(
                                                   SegmentedPNorm(ρ11, c1, C11),
                                                   SegmentedLSE(ρ12, C12),
                                                   SegmentedSum(C13),
                                                   SegmentedMax(C14)
                                                  ),
                                       Dense(W2, b2, relu)
                                      ),
                              BagModel(
                                       Dense(W3, b3, relu),
                                       Aggregation(
                                                   SegmentedPNorm(ρ21, c2, C21),
                                                   SegmentedLSE(ρ22, C22),
                                                   SegmentedSum(C23),
                                                   SegmentedMax(C24)
                                                  ),
                                       Dense(W4, b4, σ)
                                      ),
                             ), Dense(W5, b5, relu)) 
            m(tn).data
        end

        bn = BagNode(ArrayNode(z), bags3)
        bnn = BagNode(bn, bags1)
        abuilder = d -> SegmentedMeanMax(d)
        m = f64(reflectinmodel(bnn, layerbuilder, abuilder))
        @test mgradtest(params(m)...) do W1, b1, C11, C12, W2, b2, C21, C22, W3, b3
            m = BagModel(
                         BagModel(
                                  Dense(W1, b1),
                                  Aggregation(
                                              SegmentedMean(C11),
                                              SegmentedMax(C12)
                                             ),
                                  Dense(W2, b2)
                                 ),
                         Aggregation(
                                     SegmentedMean(C21),
                                     SegmentedMax(C22)
            ),
                         Dense(W3, b3)
                        )
            m(bnn).data
        end
    end
end

@testset "model aggregation grad check w.r.t. params weighted" begin
    for (bags1, bags2, bags3) in BAGS2
        layerbuilder(k) = Dense(k, 2, NNlib.relu)
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)
        w = abs.(randn(4)) .+ 0.1
        w2 = abs.(randn(4)) .+ 0.1
        w3 = abs.(randn(8)) .+ 0.1

        bn = BagNode(ArrayNode(x), bags1, w)
        abuilder = d -> SegmentedPNormLSE(d)
        m = f64(reflectinmodel(bn, layerbuilder, abuilder))
        @test mgradtest(params(m)...) do W1, b1, ρ1, c, C1, ρ2, C2, W2, b2
            m = BagModel(Dense(W1, b1, relu),
                         Aggregation(
                                     SegmentedPNorm(ρ1, c, C1),
                                     SegmentedLSE(ρ2, C2)
                                    ),
                         Dense(W2, b2, σ))
            m(bn).data
        end

        tn = ProductNode((BagNode(ArrayNode(y), bags1, w), BagNode(ArrayNode(x), bags2, w2)))
        abuilder = d -> SegmentedPNormLSESumMax(d)
        m = f64(reflectinmodel(tn, layerbuilder, abuilder))
        @test mgradtest(params(m)...) do W1, b1, ρ11, c1, C11, ρ12, C12, C13, C14,
            W2, b2, W3, b3, ρ21, c2, C21, ρ22, C22, C23, C24, W4, b4, W5, b5
            m = ProductModel((
                              BagModel(
                                       Dense(W1, b1, σ),
                                       Aggregation(
                                                   SegmentedPNorm(ρ11, c1, C11),
                                                   SegmentedLSE(ρ12, C12),
                                                   SegmentedSum(C13),
                                                   SegmentedMax(C14)
                                                  ),
                                       Dense(W2, b2, relu)
                                      ),
                              BagModel(
                                       Dense(W3, b3, relu),
                                       Aggregation(
                                                   SegmentedPNorm(ρ21, c2, C21),
                                                   SegmentedLSE(ρ22, C22),
                                                   SegmentedSum(C23),
                                                   SegmentedMax(C24)
                                                  ),
                                       Dense(W4, b4, σ)
                                      ),
                             ), Dense(W5, b5, relu)) 
            m(tn).data
        end

        bn = BagNode(ArrayNode(z), bags3, w3)
        bnn = BagNode(bn, bags1, w)
        abuilder = d -> SegmentedMeanMax(d)
        m = f64(reflectinmodel(bnn, layerbuilder, abuilder))
        @test mgradtest(params(m)...) do W1, b1, C11, C12, W2, b2, C21, C22, W3, b3
            m = BagModel(
                         BagModel(
                                  Dense(W1, b1),
                                  Aggregation(
                                              SegmentedMean(C11),
                                              SegmentedMax(C12)
                                             ),
                                  Dense(W2, b2)
                                 ),
                         Aggregation(
                                     SegmentedMean(C21),
                                     SegmentedMax(C22)
            ),
                         Dense(W3, b3)
                        )
            m(bnn).data
        end
    end

  @testset "A gradient of ProductNode with a NamedTuple " begin
    a = ProductNode((a = ArrayNode(randn(2,4)), b = ArrayNode(randn(3,4))))
    m = f64(ProductModel((a = ArrayModel(Dense(2,2)), b = ArrayModel(Dense(3,1))), Dense(3,2)))
    ps = params(m)
    gradient(() -> sum(m(a).data), ps)
    @test mgradtest(params(m)...) do W1, b1, W2, b2, W3, b3 
      m = ProductModel((a = ArrayModel(Dense(W1, b1)), b = ArrayModel(Dense(W2, b2))), Dense(W3, b3))
      sum(m(a).data)
    end
  end
end
