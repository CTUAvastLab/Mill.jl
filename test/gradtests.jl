using Flux.Tracker: gradcheck
using Mill: reflectinmodel, length2bags

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
                        Flux.data.(Flux.Tracker.gradient(f, xs...)), rtol = 1e-5, atol = 1e-5))
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
           ]

    @testset "aggregation grad check" begin
        for bags in BAGS
            w = randn(10)
            d = rand(1:1)
            x = randn(d, 10)
            for g in [
                      x -> sum(SegmentedMean()(x, bags)),
                      x -> sum(SegmentedMax()(x, bags)),
                      x -> sum(SegmentedMeanMax()(x, bags)),
                      x -> sum(SegmentedMean()(x, bags, w)),
                      x -> sum(SegmentedMax()(x, bags, w)),
                      x -> sum(SegmentedMeanMax()(x, bags, w)),
                     ]
                @test gradcheck(g, x)
            end
        end

        for bags in BAGS
            # only positive weights allowed in pnorm
            w = abs.(randn(10)) .+ 0.01
            d = rand(1:10)
            x = randn(d, 10)
            a = PNorm(d)
            @test mgradcheck(x -> sum(a(x, bags)), x)
            @test mgradcheck(x -> sum(a(x, bags, w)), x)
            @test mgradcheck(randn(d), randn(d)) do ρ, c
                n = Aggregation((PNorm(param(ρ), param(c)), SegmentedMean(), SegmentedMax()))
                sum(n(x, bags))
            end
            @test mgradcheck(randn(d), randn(d)) do ρ, c
                n = Aggregation((PNorm(param(ρ), param(c)), SegmentedMean(), SegmentedMax()))
                sum(n(x, bags, w))
            end
            @test mgradcheck(randn(d), randn(d)) do ρ, c
                n = Aggregation((SegmentedMean(), PNorm(param(ρ), param(c)), SegmentedMax()))
                sum(n(x, bags))
            end
            @test mgradcheck(randn(d), randn(d)) do ρ, c
                n = Aggregation((SegmentedMean(), PNorm(param(ρ), param(c)), SegmentedMax()))
                sum(n(x, bags, w))
            end
            @test mgradcheck(randn(d), randn(d)) do ρ, c
                n = Aggregation((SegmentedMean(), SegmentedMax(), PNorm(param(ρ), param(c))))
                sum(n(x, bags))
            end
            @test mgradcheck(randn(d), randn(d)) do ρ, c
                n = Aggregation((SegmentedMean(), SegmentedMax(), PNorm(param(ρ), param(c))))
                sum(n(x, bags, w))
            end
            @test mgradcheck(randn(d), randn(d)) do ρ, c
                n = PNorm(param(ρ), param(c))
                sum(n(x, bags))
            end
            @test mgradcheck(randn(d), randn(d)) do ρ, c
                n = PNorm(param(ρ), param(c))
                sum(n(x, bags, w))
            end
        end
    end

    @testset "model aggregation grad check w.r.t. inputs" begin
        layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)
        bags = [1:2, 3:4]
        bags2 = [1:1, 2:4]
        bags3 = [1:1, 2:2, 3:6, 7:8]

        n = ArrayNode(x)
        m = reflectinmodel(n, layerbuilder)[1]
        @test mgradcheck(x) do x
            n = ArrayNode(x)
            sum(m(n).data)
        end

        bn = BagNode(ArrayNode(x), bags)
        abuilder = d -> SegmentedPNorm(d)
        m = reflectinmodel(bn, layerbuilder, abuilder)[1]
        @test mgradcheck(x) do x
            bn = BagNode(ArrayNode(x), bags)
            sum(m(bn).data)
        end

        tn = TreeNode((ArrayNode(x), ArrayNode(y)))
        m = reflectinmodel(tn, layerbuilder)[1]
        @test mgradcheck(x, y) do x, y
            tn = TreeNode((ArrayNode(x), ArrayNode(y)))
            sum(m(tn).data)
        end

        tn = TreeNode((BagNode(ArrayNode(y), bags), BagNode(ArrayNode(x), bags2)))
        abuilder = d -> SegmentedMeanMax()
        m = reflectinmodel(tn, layerbuilder, abuilder)[1]
        @test mgradcheck(x, y) do x, y
            tn = TreeNode((BagNode(ArrayNode(y), bags), BagNode(ArrayNode(x), bags2)))
            sum(m(tn).data)
        end

        bn = BagNode(ArrayNode(z), bags3)
        bnn = BagNode(bn, bags)
        abuilder = d -> SegmentedPNormMeanMax(d)
        m = reflectinmodel(bnn, layerbuilder, abuilder)[1]
        @test mgradcheck(z) do z
            bn = BagNode(ArrayNode(z), bags3)
            bnn = BagNode(bn, bags)
            sum(m(bnn).data)
        end
    end

    @testset "model aggregation grad check w.r.t. params" begin
        layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
        x = randn(4, 4)
        y = randn(3, 4)
        z = randn(2, 8)
        bags = [1:2, 3:4]
        bags2 = [1:1, 2:4]
        bags3 = [1:1, 2:2, 3:6, 7:8]

        n = ArrayNode(x)
        m = reflectinmodel(n, layerbuilder)[1]
        @test mgradcheck(Flux.data.(Flux.params(m))...) do W, b
            m = ArrayModel(Dense(W, b, relu))
            sum(m(n).data)
        end

        bn = BagNode(ArrayNode(x), bags)
        abuilder = d -> SegmentedPNorm(d)
        m = reflectinmodel(bn, layerbuilder, abuilder)[1]
        @test mgradcheck(Flux.data.(Flux.params(m))...) do W1, b1, ρ, c, W2, b2
            m = BagModel(Dense(W1, b1, relu), PNorm(Flux.param(ρ), Flux.param(c)), Dense(W2, b2, σ))
            sum(m(bn).data)
        end

        tn = TreeNode((ArrayNode(x), ArrayNode(y)))
        m = reflectinmodel(tn, layerbuilder)[1]
        @test mgradcheck(Flux.data.(Flux.params(m))...) do W1, b1, W2, b2, W3, b3
            m = ProductModel(ArrayModel.((
                                          Dense(W1, b1, σ),
                                          Dense(W2, b2, relu)
                                         )), Dense(W3, b3, σ)) 
            sum(m(tn).data)
        end

        tn = TreeNode((BagNode(ArrayNode(y), bags), BagNode(ArrayNode(x), bags2)))
        abuilder = d -> SegmentedPNormMeanMax(d)
        m = reflectinmodel(tn, layerbuilder, abuilder)[1]
        @test mgradcheck(Flux.data.(Flux.params(m))...) do W1, b1, ρ1, c1, W2, b2, W3, b3, ρ2, c2, W4, b4, W5, b5
            m = ProductModel((
                              BagModel(
                                       Dense(W1, b1, σ),
                                       Aggregation((PNorm(Flux.param(ρ1), Flux.param(c1)), SegmentedMean(), SegmentedMax())),
                                       Dense(W2, b2, relu)
                                      ),
                              BagModel(
                                       Dense(W3, b3, relu),
                                       Aggregation((PNorm(Flux.param(ρ2), Flux.param(c2)), SegmentedMean(), SegmentedMax())),
                                       Dense(W4, b4, σ)
                                      ),
                             ), Dense(W5, b5, relu)) 
            sum(m(tn).data)
        end

        bn = BagNode(ArrayNode(z), bags3)
        bnn = BagNode(bn, bags)
        abuilder = d -> SegmentedMeanMax()
        m = reflectinmodel(bnn, layerbuilder, abuilder)[1]
        @test mgradcheck(Flux.data.(Flux.params(m))...) do W1, b1, W2, b2, W3, b3
            m = BagModel(
                         BagModel(
                                  Dense(W1, b1),
                                  SegmentedMeanMax(),
                                  Dense(W2, b2)
                                 ),
                         SegmentedMeanMax(),
                         Dense(W3, b3)
                        )
            sum(m(bnn).data)
        end
    end

    # @testset "learning basic pnorm types" begin
    # 	import Mill: p_map
    # 	model = Mill.BagModel(identity, PNorm(1), Flux.Dense(1,2))
    # 	loss(x,y) = begin
    # 		Flux.logitcrossentropy(model(x).data, Flux.onehotbatch([y], 1:2));
    # 	end
    # 	dataset = []
    # 	for t in 1:1000
    # 		p = 4
    #         c = -2 
    # 		x = rand(-1.5+c:0.01:1.5+c, 1, 2)
    # 		y = sum((abs.(x.-c)) .^ p) .^ (1/p) < 1 ? 1 : 2
    # 		push!(dataset, (Mill.BagNode(Mill.ArrayNode(x), [1:2]), y))
    # 	end
    # 
    # 	opt = Flux.ADAM(params(model))
    # 	Flux.@epochs 300 begin
    #         @show model.a.fs[1].c
    #         @show p_map(model.a.fs[1].ρ)
    # 		Flux.train!(loss, dataset, opt)
    # 		l = 0; r = 0; w = 0
    # 		for(X, y) in dataset
    # 			pred = findmax(softmax(model(X).data))[2]
    # 			if pred == y r += 1 else w += 1 end
    # 			l += loss(X, y)
    # 		end
    # 		@show l
    # 		@show r
    # 		@show w
    # 	end
    # end

end
