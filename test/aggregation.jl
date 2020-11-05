@testset "basic attributes" begin
    a1 = SegmentedMean(1:4 |> collect)
    a2 = SegmentedMean(4)
    a3 = SegmentedMeanMax(4)

    @test length(a1) == 4
    @test size(a1) == (4,)
    @test length(a2) == 4
    @test size(a2) == (4,)
    @test length(a3) == 8
    @test size(a3) == (8,)
end

@testset "vcat" begin
    # We do not implement == for models because it messes with Flux implementation
    function test_equal(a1::T, a2::T) where T <: Aggregation
        length(a1.fs) == length(a2.fs) && all(test_equal(f1, f2) for (f1, f2) in zip(a1.fs, a2.fs))
    end
    function test_equal(a1::T, a2::T) where T <: Mill.AggregationOperator
        all(getfield(a1, f) == getfield(a2, f) for f in fieldnames(T))
    end

    # Aggregation Operators
    @test test_equal(
                     vcat(SegmentedMean(ones(5)), SegmentedMean(ones(5))),
                     SegmentedMean(ones(10))
                    )
    @test test_equal(
                     vcat(SegmentedMax(ones(5)), SegmentedMax(ones(5))),
                     SegmentedMax(ones(10))
                     )
    @test test_equal(
                     vcat(SegmentedSum(ones(5)), SegmentedSum(ones(5))),
                     SegmentedSum(ones(10))
                    )
    @test test_equal(
                     vcat(SegmentedLSE(ones(5), zeros(5)), SegmentedLSE(ones(5), zeros(5))),
                     SegmentedLSE(ones(10), zeros(10))
                    )
    @test test_equal(
                     vcat(SegmentedPNorm(ones(5), zeros(5), -ones(5)),
                          SegmentedPNorm(ones(5), zeros(5), -ones(5))),
                     SegmentedPNorm(ones(10), zeros(10), -ones(10))
                    )

    # Aggregation
    @test test_equal(vcat(SegmentedMean(5)), SegmentedMean(5))
    @test test_equal(vcat(SegmentedMean(5), SegmentedMax(5)), SegmentedMeanMax(5))
end

@testset "flattening" begin
    as = [
          Aggregation(SegmentedMean(2), SegmentedMax(2)),
          Aggregation(SegmentedMean(2), Aggregation(SegmentedMax(2))),
          Aggregation((SegmentedMean(2), SegmentedMax(2))),
          Aggregation((Aggregation(SegmentedMean(2)), SegmentedMax(2))),
          Aggregation(SegmentedMeanMax(2)),
          Aggregation(Aggregation(SegmentedMeanMax(2))),
          Aggregation((SegmentedMeanMax(2),)),
          SegmentedMeanMax(2)
         ]

    @test all(length.(as) .== 4)
    @test all(a -> all(f -> length(f) == 2, a.fs), as)
    @test all(isa.(getindex.(as, 1), SegmentedMean))
    @test all(isa.(getindex.(as, 2), SegmentedMax))
end

@testset "basic aggregation functionality" begin
    W = [1, 1/2, 1/2, 1/8, 1/3, 13/24] |> f32
    X = reshape(1:12, 2, 6) |> f32
    bags = BAGS[1]
    baglengths = log.(1.0 .+ [1.0 2.0 3.0]) |> f32
    @assert baglengths == log.(1 .+ length.(bags)') |> f32
    @test SegmentedMean(2)(X, bags) ≈ [1.0 4.0 9.0; 2.0 5.0 10.0; baglengths]
    @test SegmentedSum(2)(X, bags) ≈ [length.(bags)' .* SegmentedMean(2)(X, bags)[1:2, :]; baglengths]
    @test SegmentedMax(2)(X, bags) ≈ [1.0 5.0 11.0; 2.0 6.0 12.0; baglengths]
    @test SegmentedMeanMax(2)(X, bags) ≈ vcat(SegmentedMean(2)(X, bags)[1:2, :],
                                              SegmentedMax(2)(X, bags)[1:2, :], baglengths)
    @test SegmentedMean(2)(X, bags, W) ≈ [1.0 4.0 236/24; 2.0 5.0 260/24; baglengths]
    bagnorms = [bagnorm(W, b) for b in bags]
    @test SegmentedSum(2)(X, bags, W) ≈ [bagnorms' .* SegmentedMean(2)(X, bags, W)[1:2, :]; baglengths]
    @test SegmentedMax(2)(X, bags, W) ≈ SegmentedMax(2)(X, bags)
end

@testset "matrix weights" begin
    W = abs.(rand(Float32, 6))
    W_mat = vcat(W, 2*W)
    X = reshape(1:12, 2, 6) |> f32
    for bags in BAGS
        @test SegmentedSum(2)(X, bags, W_mat) ≈ SegmentedSum(2)(X, bags, W)
        @test SegmentedMean(2)(X, bags, W_mat) ≈ SegmentedMean(2)(X, bags, W)
        @test SegmentedMax(2)(X, bags, W_mat) ≈ SegmentedMax(2)(X, bags, W)
    end
end

@testset "pnorm functionality" begin
    dummy = randn(2)
    for t = 1:10
        a, b, c, d, ρ1, ρ2, c1, c2, w1, w2 = randn(10)
        p1, p2 = p_map(ρ1), p_map(ρ2)
        w1 = abs(w1)
        w2 = abs(w2)
        bags = ScatteredBags([[1,2]])
        agg = SegmentedPNorm(dummy, [ρ1, ρ2], [c1, c2])
        @test agg([a b; c d], bags) ≈ [
                                     (1/2*(abs(a-c1)^p1 + abs(b-c1)^p1))^(1/p1);
                                     (1/2*(abs(c-c2)^p2 + abs(d-c2)^p2))^(1/p2)
                                    ]
        @test agg([a b; c d], bags) ≈ [
                                     (1/2*(abs(a-c1)^p1 + abs(b-c1)^p1))^(1/p1);
                                     (1/2*(abs(c-c2)^p2 + abs(d-c2)^p2))^(1/p2)
                                    ]
        @test agg([a b; c d], bags, [w1, w2]) ≈ [
                                               (1/(w1+w2)*(w1*abs(a-c1)^p1 + w2*abs(b-c1)^p1))^(1/p1);
                                               (1/(w1+w2)*(w1*abs(c-c2)^p2 + w2*abs(d-c2)^p2))^(1/p2)
                                              ]
        for bags in BAGS
            X = randn(2, 6)
            agg = SegmentedPNorm(dummy, inv_p_map([1+1e-16, 1+1e-16]), [0.0, 0.0])
            @test agg(X, bags) ≈ SegmentedMean(dummy)(abs.(X), bags)
            agg = SegmentedPNorm(dummy, inv_p_map([2.0, 2.0]), [0.0, 0.0])
            @test agg(X, bags) ≈ hcat([sqrt.(sum(X[:, b] .^ 2, dims=2) ./ length(b)) for b in bags]...)
        end
    end
end

@testset "lse functionality" begin
    dummy = randn(2)
    for t = 1:10
        a, b, c, d, ρ1, ρ2 = randn(6)
        r1, r2 = r_map(ρ1), r_map(ρ2)
        @test SegmentedLSE(dummy, [ρ1, ρ2])([a b; c d], ScatteredBags([[1,2]])) ≈ [
                                                                               1/r1*log(1/2*(exp(a*r1)+exp(b*r1)));
                                                                               1/r2*log(1/2*(exp(c*r2)+exp(d*r2)))
                                                                              ]
        for bags in BAGS
            X = randn(2, 6)
            W = abs.(randn(6))
            r1, r2 = randn(2)
            # doesn't use weights
            @test all(SegmentedLSE(dummy, [ρ1, ρ2])(X, bags) .== SegmentedLSE(dummy, [ρ1, ρ2])(X, bags, W))
            # the bigger value of r, the closer we are to the real maximum
            @test isapprox(SegmentedLSE(dummy, [100.0, 100.0])(X, bags), SegmentedMax(dummy)(X, bags), atol=0.1)
        end
    end
end

@testset "pnorm numerical stability" begin
    k, d = 10, 5
    dummy, c = randn(d), zeros(d)
    b = AlignedBags([1:k])
    ρ1 = inv_p_map(ones(d))
    ρ2 = randn(d)
    p2 = p_map(ρ2)
    Z = 1e5 .+ 1e3 .* randn(d, k)
    W = abs.(randn(k)) .+ 1e-2
    for X in [Z, -Z, randn(d, k)]
        @test SegmentedPNorm(dummy, ρ1, c)(X, b) ≈ sum(abs.(X); dims=2) ./ k
        @test SegmentedPNorm(dummy, ρ1, c)(X, b, W) ≈ sum(W' .* abs.(X); dims=2) / sum(W)
        @test SegmentedPNorm(dummy, ρ1, c)(X, b, W) ≈ sum(W' .* abs.(X); dims=2) / sum(W)
        for i in 1:k
            @test SegmentedPNorm(dummy, ρ1, c)(repeat(X[:, i], 1, k), b) ≈ abs.(X[:, i])
            @test SegmentedPNorm(dummy, ρ2, c)(repeat(X[:, i], 1, k), b) ≈ abs.(X[:, i])
            @test SegmentedPNorm(dummy, ρ1, c)(repeat(X[:, i], 1, k), b, W) ≈ abs.(X[:, i])
            @test SegmentedPNorm(dummy, ρ1, c)(repeat(X[:, i], 1, k), b, W) ≈ abs.(X[:, i])
            @test SegmentedPNorm(dummy, ρ2, c)(repeat(X[:, i], 1, k), b, W) ≈ abs.(X[:, i])
            @test SegmentedPNorm(dummy, ρ2, c)(repeat(X[:, i], 1, k), b, W) ≈ abs.(X[:, i])
        end
    end
end

@testset "lse numerical stability" begin
    k, d = 10, 5
    dummy = randn(d)
    b = AlignedBags([1:k])
    ρ1 = inv_r_map(1e15 .+ 1e5 .* randn(d))
    ρ2 = inv_r_map(abs.(1e5 .* randn(d)))
    Z = 1e5 .+ 1e3 .* randn(d, k)
    W = abs.(randn(k)) .+ 1e-2
    for X in [Z, -Z, randn(d, k)]
        @test SegmentedLSE(dummy, ρ1)(X, b) ≈ maximum(X; dims=2)
        # doesn't use weights
        @test SegmentedLSE(dummy, ρ1)(X, b, W) ≈ maximum(X; dims=2)

        # implementation immune to underflow not available yet
        @test_skip @test SegmentedLSE(dummy, ρ2)(X, b) ≈ sum(X; dims=2) ./ k
        # doesn't use weights
        @test_skip @test SegmentedLSE(dummy, ρ2)(X, b, W) ≈ sum(X; dims=2) ./ k

        for i in 1:k
            @test SegmentedLSE(dummy, randn(d))(repeat(X[:, i], 1, k), b) ≈ X[:, i]
            # doesn't use weights
            @test SegmentedLSE(dummy, randn(d))(repeat(X[:, i], 1, k), b, W) ≈ X[:, i]
        end
    end
end

@testset "missing values" begin
    dummy = randn(2)
    ψ = randn(2)
    for bags in [AlignedBags([0:-1]), AlignedBags([0:-1, 0:-1, 0:-1])]
        @test SegmentedMean(ψ)(missing, bags) == repeat(ψ, 1, length(bags))
        @test SegmentedSum(ψ)(missing, bags) == repeat(ψ, 1, length(bags))
        @test SegmentedMax(ψ)(missing, bags) == repeat(ψ, 1, length(bags))
        @test SegmentedLSE(ψ, dummy)(missing, bags) == repeat(ψ, 1, length(bags))
        @test SegmentedPNorm(ψ, dummy, dummy)(missing, bags) == repeat(ψ, 1, length(bags)) 
    end

    # default values ψ are indeed filled in
    for bags in vcat(BAGS2)
        idcs = isempty.(bags.bags)
        l = maximum(maximum.(bags.bags[.!idcs]))
        X = randn(2, l)
        @test SegmentedMean(ψ)(X, bags)[:, idcs] == repeat(ψ, 1, sum(idcs))
        @test SegmentedSum(ψ)(X, bags)[:, idcs] == repeat(ψ, 1, sum(idcs))
        @test SegmentedMax(ψ)(X, bags)[:, idcs] == repeat(ψ, 1, sum(idcs))
        @test SegmentedLSE(ψ, dummy)(X, bags)[:, idcs] == repeat(ψ, 1, sum(idcs))
        @test SegmentedPNorm(ψ, dummy, dummy)(X, bags)[:, idcs] == repeat(ψ, 1, sum(idcs))
    end
end

@testset "bagcount switch" begin
    X = reshape(1:12, 2, 6) |> f32
    d = 2
    bags = BAGS[1]
    baglengths = log.(1 .+ [1.0 2.0 3.0]) |> f32

    function test_count(a)
        Mill.bagcount!(false)
        a1 = a(X, bags)
        Mill.bagcount!(true)
        a2 = a(X, bags)
        @test [a1; baglengths] == a2
    end

    test_count(SegmentedSum(2))
    test_count(SegmentedMean(2))
    test_count(SegmentedMax(2))
    test_count(SegmentedPNorm(2))
    test_count(SegmentedLSE(2))
    test_count(SegmentedPNormLSE(2))
    test_count(SegmentedMeanMaxPNormLSE(2))
end

# we use Float64 to compute precise gradients

@testset "grad check w.r.t. input" begin
    for bags in BAGS2
        d = rand(1:20)
        x = randn(d, 10)
        w = abs.(randn(size(x, 2))) .+ 0.1
        w_mat = abs.(randn(size(x))) .+ 0.1

        # generate all combinations of aggregations
        anames = ["Sum", "Mean", "Max", "PNorm", "LSE"]
        for idxs in powerset(collect(1:length(anames)))
            1 ≤ length(idxs) ≤ 3 || continue

            s = Symbol("Segmented", anames[idxs]...)
            a = @eval $s($d) |> f64
            @test gradtest(x) do x
                a(x, bags)
            end
            @test gradtest(x) do x
                a(x, bags, w)
            end
            @test gradtest(x) do x
                a(x, bags, w_mat)
            end
        end
    end
end

@testset "grad check w.r.t. agg params" begin
    # r_map and p_map are stable
    @test first(gradient(softplus, 10000)) ≈ σ(10000) ≈ 1.0
    @test first(gradient(softplus, -10000)) ≈ σ(-10000) ≈ 0

    fs = [:SegmentedSum, :SegmentedMean, :SegmentedMax, :SegmentedPNorm, :SegmentedLSE]
    params = [(:ψ1,), (:ψ2,), (:ψ3,), (:ψ4, :ρ1, :c), (:ψ5, :ρ2)]

    for idxs in powerset(collect(1:length(fs)))
        1 ≤ length(idxs) <= 2 || continue

        d = rand(1:20)
        x = randn(d, 0)
        as = []; cs = []; rs = []
        for (f, ps) in zip(fs[idxs], params[idxs])
            push!(rs, fill(:(randn($d)), length(ps))...)
            push!(as, ps...)
            push!(cs, Expr(:call, f, ps...))
        end
        @eval begin
            @test gradtest($(map(eval, rs)...)) do $(as...)
                a = Aggregation($(cs...))
                a(missing, ScatteredBags([Int[], Int[]]))
            end
            @test gradtest($(map(eval, rs)...)) do $(as...)
                a = Aggregation($(cs...))
                a(missing, AlignedBags([0:-1]), nothing)
            end
            @test gradtest($(map(eval, rs)...)) do $(as...)
                a = Aggregation($(cs...))
                a($x, ScatteredBags([Int[]]))
            end
            @test gradtest($(map(eval, rs)...)) do $(as...)
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
                @test gradtest($(map(eval, rs)...)) do $(as...)
                    a = Aggregation($(cs...))
                    a($x, $bags)
                end
                @test gradtest($(map(eval, rs)...)) do $(as...)
                    a = Aggregation($(cs...))
                    a($x, $bags, $w)
                end
                @test gradtest($(map(eval, rs)...)) do $(as...)
                    a = Aggregation($(cs...))
                    a($x, $bags, $w_mat)
                end
            end
        end
    end
end

@testset "derivative w.r.t weights" begin
    for bags in BAGS2
        d = rand(1:20)
        x = randn(d, 10)
        w = abs.(randn(size(x, 2))) .+ 0.1
        w_mat = abs.(randn(size(x))) .+ 0.1

        a1 = SegmentedSum(d) |> f64
        a2 = SegmentedMean(d) |> f64
        a3 = SegmentedMax(d) |> f64
        a4 = SegmentedPNorm(d) |> f64
        a5 = SegmentedLSE(d) |> f64
        for g in [
                  w -> a1(x, bags, w),
                  w -> a2(x, bags, w),
                  w -> a3(x, bags, w),
                  w -> a5(x, bags, w)
                 ]
            @test gradtest(g, w)
        end
        for g in [
                  w_mat -> a1(x, bags, w_mat),
                  w_mat -> a2(x, bags, w_mat),
                  w_mat -> a3(x, bags, w_mat),
                  w_mat -> a5(x, bags, w_mat)
                 ]
            @test gradtest(g, w_mat)
        end
        # for g in [
        #           w -> a4(x, bags, w_mat)
        #          ]
        #     # NOT IMPLEMENTED YET
        #     @test_throws Exception gradtest(g, w_mat)
        # end
        # for g in [
        #           w -> a4(x, bags, w_mat)
        #          ]
        #     # NOT IMPLEMENTED YET
        #     @test_throws Exception gradtest(g, w_mat)
        # end
    end
end
