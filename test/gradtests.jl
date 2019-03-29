using Mill: reflectinmodel, length2bags
using Combinatorics

# working with tracked output - now it is possible to test whole models
# suitable for situations where the output of the model is TrackedArray
# native gradcheck in Flux tests only operations and not models
function mngradient(f, xs::AbstractArray...)
    grads = zero.(xs)
    for (x, Δ) in zip(xs, grads), i in 1:length(x)
        δ = sqrt(eps())
        tmp = x[i]
        x[i] = tmp - δ/2
        y1 = Flux.data(f(xs...))
        x[i] = tmp + δ/2
        y2 = Flux.data(f(xs...))
        x[i] = tmp
        Δ[i] = (y2-y1)/δ
    end
    return grads
end

function mgradcheck(f, xs...)
    ret = all(isapprox.(mngradient(f, xs...),
                        Flux.data.(Flux.Tracker.gradient(f, xs...)), rtol = 1e-4, atol = 1e-4))
    if !ret
        @show mngradient(f, xs...)
        @show Flux.Tracker.gradient(f, xs...)
    end
    return ret
end

let
    BAGS = [
            length2bags([1 for _ in 1:10]),
            length2bags([2 for _ in 1:5]),
            length2bags([5, 5]),
            length2bags([10]),
            length2bags([3, 4, 3]),
            ScatteredBags([collect(1:3), collect(7:10), collect(4:6)]),
            ScatteredBags([collect(1:10)])
           ]

    @testset "aggregation grad check w.r.t. input" begin
        for bags in BAGS
            # only positive weights allowed in pnorm and lse
            d = rand(1:20)
            x = randn(d, 10)
            w = abs.(randn(size(x, 2))) .+ 0.01

            # generate all combinations of aggregations
            as = []
            names = ["Mean", "Max", "PNorm", "LSE"]
            for idxs in powerset(collect(1:length(names)))
                !isempty(idxs) || continue
                for p in permutations(idxs)
                    s = Symbol("Segmented", names[p]...)
                    push!(as, @eval $s($d))
                end
            end
            # both weighted and unweighted versions
            for g in vcat(map(a -> x->sum(a(x, bags)), as),
                          map(a -> x->sum(a(x, bags, w)), as))
                # @test mgradcheck(g, x)
            end
        end
    end

    @testset "aggregation grad check w.r.t. agg params" begin
        # for bags in BAGS[1:5]
        for bags in BAGS
            # only positive weights allowed in pnorm and lse
            d = rand(1:20)
            x = randn(d, 10)
            w = abs.(randn(size(x, 2))) .+ 0.01
            
            # Random.seed!(0)
            # d = 1
            # x = reshape(collect(1.0:10.0), 1, 10)
            # w = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]

            fs = [:SegmentedMax, :SegmentedMean, :SegmentedPNorm, :SegmentedLSE]
            # TODO test Cs with empty bags
            params = [(:C1,), (:C2,), (:ρ, :c, :C3), (:p, :C4)]

            fs = fs[3:3]
            params = params[3:3]

            for idxs in powerset(collect(1:length(fs)))
                !isempty(idxs) || continue;
                rs = []; as = []; cs = []; 
                for (f, ps) in zip(fs[idxs], params[idxs])
                    push!(rs, (:(randn($d)) for _ in ps)...)
                    push!(as, ps...)
                    push!(cs, Expr(:call, f, map(p -> :(param($p)), ps)...))
                end
                @eval f = (x, bags) -> mgradcheck($(map(eval, rs)...)) do $(as...)
                    n = Aggregation($(cs...))
                    sum(n(x, bags))
                end
                @test f(x, bags)
                @eval g = (x, bags, w) -> mgradcheck($(map(eval, rs)...)) do $(as...)
                    n = Aggregation($(cs...))
                    sum(n(x, bags, w))
                end
                # @test g(x, bags, w)
            end
        end
    end

    @testset "aggregation grad check w.r.t. params of aggregation" begin
        # for bags in BAGS
        #     # only positive weights allowed in pnorm and lse
        #     w = abs.(randn(Float32, 10)) .+ Float32(0.01)
        #     d = rand(1:10)
        #     x = randn(Float32, d, 10)
        #     a = PNorm(d)
        #     b = LSE(d)
        #     @test mgradcheck(x -> sum(a(x, bags)), x)
        #     @test mgradcheck(x -> sum(b(x, bags)), x)
        #     @test mgradcheck(x -> sum(a(x, bags, w)), x)
        #     @test mgradcheck(x -> sum(b(x, bags, w)), x)

        #     # test on gradients w.r.t. params
        #     fs = [:PNorm, :LSE, :SegmentedMean, :SegmentedMax]
        #     # TODO
        #     # params = [(:C, :ρ, :c), (:C, :p,), (:C), (:C)]
        #     params = [(:ρ, :c), (:p,), (:C), (:C)]
        #     for idxs in powerset(collect(1:length(fs)))
        #         1 in idxs || 2 in idxs || continue
        #         rs = []; as = []; cs = []; 
        #         for (f, ps) in zip(fs[idxs], params[idxs])
        #             push!(rs, (:(randn(Float32, $d)) for _ in ps)...)
        #             push!(as, ps...)
        #             push!(cs, Expr(:call, f, map(p -> :(param($p)), ps)...))
        #         end
        #         @eval f = (x, bags) -> mgradcheck($(map(eval, rs)...)) do $(as...)
        #             n = Aggregation(tuple($(cs...)))
        #             sum(n(x, bags))
        #         end
        #         @test f(x, bags)
        #         @eval g = (x, bags, w) -> mgradcheck($(map(eval, rs)...)) do $(as...)
        #             n = Aggregation(tuple($(cs...)))
        #             sum(n(x, bags, w))
        #         end
        #         @test g(x, bags, w)
        #     end
        # end
    end


end

#     @testset "derivative w.r.t weights in aggregations" begin
#         for bags in BAGS
#             w = randn(Float32, 10)
#             d = rand(1:10)
#             x = randn(Float32, d, 10)
#             for g in [
#                       w -> sum(SegmentedMean(d)(x, bags, w)),
#                       w -> sum(SegmentedMax(d)(x, bags, w))
#                      ]
#                 @test gradcheck(g, w)
#             end
#             @test mgradcheck(x, w) do x, w
#                 sum(SegmentedMean(d)(x, bags, w))
#             end
#             @test mgradcheck(x, w) do x, w
#                 sum(SegmentedMax(d)(x, bags, w))
#             end
#         end
#     end

#     @testset "model aggregation grad check w.r.t. inputs" begin
#         layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
#         x = randn(Float32, 4, 4)
#         y = randn(Float32, 3, 4)
#         z = randn(Float32, 2, 8)
#         bags = AlignedBags([1:2, 3:4])
#         bags2 = AlignedBags([1:1, 2:4])
#         bags3 = AlignedBags([1:1, 2:2, 3:6, 7:8])

#         n = ArrayNode(x)
#         m = reflectinmodel(n, layerbuilder)[1]
#         @test mgradcheck(x) do x
#             n = ArrayNode(x)
#             sum(m(n).data)
#         end

#         bn = BagNode(ArrayNode(x), bags)
#         abuilder = d -> SegmentedPNormLSE(d)
#         m = reflectinmodel(bn, layerbuilder)[1]
#         @test mgradcheck(x) do x
#             bn = BagNode(ArrayNode(x), bags)
#             sum(m(bn).data)
#         end

#         tn = TreeNode((ArrayNode(x), ArrayNode(y)))
#         m = reflectinmodel(tn, layerbuilder)[1]
#         @test mgradcheck(x, y) do x, y
#             tn = TreeNode((ArrayNode(x), ArrayNode(y)))
#             sum(m(tn).data)
#         end

#         tn = TreeNode((BagNode(ArrayNode(y), bags), BagNode(ArrayNode(x), bags2)))
#         abuilder = d -> SegmentedMeanMax(d)
#         m = reflectinmodel(tn, layerbuilder, abuilder)[1]
#         @test mgradcheck(x, y) do x, y
#             tn = TreeNode((BagNode(ArrayNode(y), bags), BagNode(ArrayNode(x), bags2)))
#             sum(m(tn).data)
#         end

#         bn = BagNode(ArrayNode(z), bags3)
#         bnn = BagNode(bn, bags)
#         abuilder = d -> SegmentedPNormLSEMeanMax(d)
#         m = reflectinmodel(bnn, layerbuilder, abuilder)[1]
#         @test mgradcheck(z) do z
#             bn = BagNode(ArrayNode(z), bags3)
#             bnn = BagNode(bn, bags)
#             sum(m(bnn).data)
#         end
#     end

#     @testset "model aggregation grad check w.r.t. inputs weighted" begin
#         layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
#         x = randn(Float32, 4, 4)
#         y = randn(Float32, 3, 4)
#         z = randn(Float32, 2, 8)
#         w = abs.(randn(Float32, 4)) .+ 0.01
#         w2 = abs.(randn(Float32, 4)) .+ 0.01
#         w3 = abs.(randn(Float32, 8)) .+ 0.01
#         bags = AlignedBags([1:2, 3:4])
#         bags2 = AlignedBags([1:1, 2:4])
#         bags3 = AlignedBags([1:1, 2:2, 3:6, 7:8])

#         bn = BagNode(ArrayNode(x), bags, w)
#         abuilder = d -> SegmentedPNormLSE(d)
#         m = reflectinmodel(bn, layerbuilder)[1]
#         @test mgradcheck(x) do x
#             bn = BagNode(ArrayNode(x), bags, w)
#             sum(m(bn).data)
#         end

#         tn = TreeNode((BagNode(ArrayNode(y), bags, w), BagNode(ArrayNode(x), bags2, w2)))
#         abuilder = d -> SegmentedMeanMax(d)
#         m = reflectinmodel(tn, layerbuilder, abuilder)[1]
#         @test mgradcheck(x, y) do x, y
#             tn = TreeNode((BagNode(ArrayNode(y), bags, w), BagNode(ArrayNode(x), bags2, w2)))
#             sum(m(tn).data)
#         end

#         bn = BagNode(ArrayNode(z), bags3, w3)
#         bnn = BagNode(bn, bags)
#         abuilder = d -> SegmentedPNormLSEMeanMax(d)
#         m = reflectinmodel(bnn, layerbuilder, abuilder)[1]
#         @test mgradcheck(z) do z
#             bn = BagNode(ArrayNode(z), bags3, w3)
#             bnn = BagNode(bn, bags, w)
#             sum(m(bnn).data)
#         end
#     end

#     @testset "model aggregation grad check w.r.t. params" begin
#         layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
#         x = randn(Float32, 4, 4)
#         y = randn(Float32, 3, 4)
#         z = randn(Float32, 2, 8)
#         bags = [1:2, 3:4]
#         bags2 = [1:1, 2:4]
#         bags3 = [1:1, 2:2, 3:6, 7:8]

#         n = ArrayNode(x)
#         m = reflectinmodel(n, layerbuilder)[1]
#         @test mgradcheck(Flux.data.(Flux.params(m))...) do W, b
#             m = ArrayModel(Dense(W, b, relu))
#             sum(m(n).data)
#         end

#         bn = BagNode(ArrayNode(x), bags)
#         abuilder = d -> SegmentedPNormLSE(d)
#         m = reflectinmodel(bn, layerbuilder, abuilder)[1]
#         @test mgradcheck(Flux.data.(Flux.params(m))...) do W1, b1, ρ, c, p, W2, b2
#             m = BagModel(Dense(W1, b1, relu),
#                          Aggregation((PNorm(Flux.param(ρ), Flux.param(c)), LSE(param(p)))),
#                          Dense(W2, b2, σ))
#             sum(m(bn).data)
#         end

#         tn = TreeNode((ArrayNode(x), ArrayNode(y)))
#         m = reflectinmodel(tn, layerbuilder)[1]
#         @test mgradcheck(Flux.data.(Flux.params(m))...) do W1, b1, W2, b2, W3, b3
#             m = ProductModel(ArrayModel.((
#                                           Dense(W1, b1, σ),
#                                           Dense(W2, b2, relu)
#                                          )), Dense(W3, b3, σ)) 
#             sum(m(tn).data)
#         end

#         tn = TreeNode((BagNode(ArrayNode(y), bags), BagNode(ArrayNode(x), bags2)))
#         abuilder = d -> SegmentedPNormLSEMeanMax(d)
#         m = reflectinmodel(tn, layerbuilder, abuilder)[1]
#         @test mgradcheck(Flux.data.(Flux.params(m))...) do W1, b1, ρ1, c1, p1, W2, b2, W3, b3, ρ2, c2, p2, W4, b4, W5, b5
#             m = ProductModel((
#                               BagModel(
#                                        Dense(W1, b1, σ),
#                                        Aggregation((
#                                                     PNorm(Flux.param(ρ1), Flux.param(c1)),
#                                                     LSE(param(p1)),
#                                                     SegmentedMean(d),
#                                                     SegmentedMax(d))),
#                                        Dense(W2, b2, relu)
#                                       ),
#                               BagModel(
#                                        Dense(W3, b3, relu),
#                                        Aggregation((
#                                                     PNorm(Flux.param(ρ2), Flux.param(c2)),
#                                                     LSE(param(p2)),
#                                                     SegmentedMean(d),
#                                                     SegmentedMax(d))),
#                                        Dense(W4, b4, σ)
#                                       ),
#                              ), Dense(W5, b5, relu)) 
#             sum(m(tn).data)
#         end

#         bn = BagNode(ArrayNode(z), bags3)
#         bnn = BagNode(bn, bags)
#         abuilder = d -> SegmentedMeanMax(d)
#         m = reflectinmodel(bnn, layerbuilder, abuilder)[1]
#         @test mgradcheck(Flux.data.(Flux.params(m))...) do W1, b1, W2, b2, W3, b3
#             m = BagModel(
#                          BagModel(
#                                   Dense(W1, b1),
#                                   SegmentedMeanMax(d),
#                                   Dense(W2, b2)
#                                  ),
#                          SegmentedMeanMax(d),
#                          Dense(W3, b3)
#                         )
#             sum(m(bnn).data)
#         end
#     end

#     @testset "model aggregation grad check w.r.t. params weighted" begin
#         layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
#         x = randn(Float32, 4, 4)
#         y = randn(Float32, 3, 4)
#         z = randn(Float32, 2, 8)
#         w = abs.(randn(Float32, 4)) .+ 0.01
#         w2 = abs.(randn(Float32, 4)) .+ 0.01
#         w3 = abs.(randn(Float32, 8)) .+ 0.01
#         bags = [1:2, 3:4]
#         bags2 = [1:1, 2:4]
#         bags3 = [1:1, 2:2, 3:6, 7:8]

#         bn = BagNode(ArrayNode(x), bags, w)
#         abuilder = d -> SegmentedPNormLSE(d)
#         m = reflectinmodel(bn, layerbuilder, abuilder)[1]
#         @test mgradcheck(Flux.data.(Flux.params(m))...) do W1, b1, ρ, c, p, W2, b2
#             m = BagModel(Dense(W1, b1, relu),
#                          Aggregation((PNorm(Flux.param(ρ), Flux.param(c)), LSE(param(p)))),
#                          Dense(W2, b2, σ))
#             sum(m(bn).data)
#         end

#         tn = TreeNode((BagNode(ArrayNode(y), bags, w), BagNode(ArrayNode(x), bags2, w2)))
#         abuilder = d -> SegmentedPNormLSEMeanMax(d)
#         m = reflectinmodel(tn, layerbuilder, abuilder)[1]
#         @test mgradcheck(Flux.data.(Flux.params(m))...) do W1, b1, ρ1, c1, p1, W2, b2, W3, b3, ρ2, c2, p2, W4, b4, W5, b5
#             m = ProductModel((
#                               BagModel(
#                                        Dense(W1, b1, σ),
#                                        Aggregation((
#                                                     PNorm(Flux.param(ρ1), Flux.param(c1)),
#                                                     LSE(param(p1)),
#                                                     SegmentedMean(),
#                                                     SegmentedMax())),
#                                        Dense(W2, b2, relu)
#                                       ),
#                               BagModel(
#                                        Dense(W3, b3, relu),
#                                        Aggregation((
#                                                     PNorm(Flux.param(ρ2), Flux.param(c2)),
#                                                     LSE(param(p2)),
#                                                     SegmentedMean(),
#                                                     SegmentedMax())),
#                                        Dense(W4, b4, σ)
#                                       ),
#                              ), Dense(W5, b5, relu)) 
#             sum(m(tn).data)
#         end

#         bn = BagNode(ArrayNode(z), bags3, w3)
#         bnn = BagNode(bn, bags, w)
#         abuilder = d -> SegmentedMeanMax()
#         m = reflectinmodel(bnn, layerbuilder, abuilder)[1]
#         @test mgradcheck(Flux.data.(Flux.params(m))...) do W1, b1, W2, b2, W3, b3
#             m = BagModel(
#                          BagModel(
#                                   Dense(W1, b1),
#                                   SegmentedMeanMax(),
#                                   Dense(W2, b2)
#                                  ),
#                          SegmentedMeanMax(),
#                          Dense(W3, b3)
#                         )
#             sum(m(bnn).data)
#         end
#     end
     
# end
