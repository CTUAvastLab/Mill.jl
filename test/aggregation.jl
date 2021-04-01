@testset "basic attributes" begin
    a1 = SegmentedMean(1:4 |> collect)
    a2 = SegmentedMean(4)
    a3 = meanmax_aggregation(4)

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
    @test test_equal(vcat(mean_aggregation(5), max_aggregation(5)), meanmax_aggregation(5))
end

@testset "flattening" begin
    as = [
          Aggregation(SegmentedMean(2), SegmentedMax(2)),
          Aggregation(SegmentedMean(2), Aggregation(SegmentedMax(2))),
          Aggregation((SegmentedMean(2), SegmentedMax(2))),
          Aggregation((Aggregation(SegmentedMean(2)), SegmentedMax(2))),
          Aggregation(meanmax_aggregation(2)),
          Aggregation(Aggregation(meanmax_aggregation(2))),
          Aggregation((meanmax_aggregation(2),)),
          meanmax_aggregation(2)
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
    @test mean_aggregation(2)(X, bags) ≈ [1.0 4.0 9.0; 2.0 5.0 10.0; baglengths]
    @test sum_aggregation(2)(X, bags) ≈ [length.(bags)' .* mean_aggregation(2)(X, bags)[1:2, :]; baglengths]
    @test max_aggregation(2)(X, bags) ≈ [1.0 5.0 11.0; 2.0 6.0 12.0; baglengths]
    @test meanmax_aggregation(2)(X, bags) ≈ vcat(mean_aggregation(2)(X, bags)[1:2, :],
                                              max_aggregation(2)(X, bags)[1:2, :], baglengths)
    @test mean_aggregation(2)(X, bags, W) ≈ [1.0 4.0 236/24; 2.0 5.0 260/24; baglengths]
    bagnorms = [_bagnorm(W, b) for b in bags]
    @test sum_aggregation(2)(X, bags, W) ≈ [bagnorms' .* mean_aggregation(2)(X, bags, W)[1:2, :]; baglengths]
    @test max_aggregation(2)(X, bags, W) ≈ max_aggregation(2)(X, bags)
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
            @test SegmentedLSE(dummy, [ρ1, ρ2])(X, bags) == SegmentedLSE(dummy, [ρ1, ρ2])(X, bags, W)
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

@testset "type stability of the output w.r.t. missing" begin
    X1 = [1.0 2.0; 3.0 4.0] |> f32
    X2 = [1.0 missing; 3.0 4.0] |> Matrix{Maybe{Float32}}
    X3 = [1.0 2.0; 3.0 4.0] |> Matrix{Maybe{Float32}}
    X4 = missing

    b1 = bags([1:2])
    b2 = bags([0:-1])

    a = all_aggregations(Float32, 2)
    @test eltype(a(X1, b1)) === Float32
    @test eltype(a(X2, b1)) === Maybe{Float32}
    @test eltype(a(X3, b1)) === Maybe{Float32}
    @test eltype(a(X4, b2)) === Float32

    a = all_aggregations(Float32, 10)
    for b in BAGS2
        X = rand(Float32, 10, 100)
        w = abs.(randn(Float32, size(X, 2))) .+ 0.1
        w_mat = abs.(randn(Float32, size(X))) .+ 0.1
        @inferred a(X, b)
        @inferred a(X, b, w |> f32)
        @inferred a(X, b, w_mat |> f32)
     end
end

@testset "type stability of the output w.r.t. precision" begin
    X32 = [1.0 2.0; 3.0 4.0] |> f32
    X64 = [1.0 2.0; 3.0 4.0]

    b = bags([1:2])

    a32 = all_aggregations(Float32, 2)
    a64 = all_aggregations(Float64, 2)
    @test eltype(a32(X32, b)) === Float32
    @test eltype(a64(X64, b)) === Float64
    @test eltype(a32(X64, b)) === Float64
    @test eltype(a64(X32, b)) === Float64

    a32 = all_aggregations(Float32, 10)
    a64 = all_aggregations(Float64, 10)
    for b in BAGS2
        X32 = rand(Float32, 10, 100)
        X64 = rand(10, 100)
        w32 = abs.(randn(Float32, size(X32, 2))) .+ 0.1f0
        w64 = abs.(randn(Float64, size(X64, 2))) .+ 0.1
        w_mat32 = abs.(randn(Float32, size(X32))) .+ 0.1f0
        w_mat64 = abs.(randn(Float64, size(X64))) .+ 0.1
        for a in [a32, a64]
            for (X, w, w_mat) in [(X32, w32, w_mat32), (X64, w64, w_mat64)]
                @inferred a(X, b)
                @inferred a(X, b, w)
                @inferred a(X, b, w_mat)
            end
        end
     end
end

@testset "r_map and p_map are stable" begin
    @test first(gradient(softplus, 10000)) ≈ σ(10000) ≈ 1.0
    @test first(gradient(softplus, -10000)) ≈ σ(-10000) ≈ 0
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

    test_count(sum_aggregation(2))
    test_count(mean_aggregation(2))
    test_count(max_aggregation(2))
    test_count(pnorm_aggregation(2))
    test_count(lse_aggregation(2))
    test_count(pnormlse_aggregation(2))
    test_count(meanmaxpnormlse_aggregation(2))
end

@testset "aggregation grad check w.r.t. input" begin
    for bags in BAGS2
        d = rand(1:20)
        x = randn(d, 10)
        w = abs.(randn(size(x, 2))) .+ 0.1
        w_mat = abs.(randn(size(x))) .+ 0.1
        a = all_aggregations(Float64, d)
        @test gradtest(x -> a(x, bags), x)
        @test gradtest(x -> a(x, bags, w), x)
        @test gradtest(x -> a(x, bags, w_mat), x)
    end
end

@testset "aggregation gradcheck w.r.t weights" begin
    for bags in BAGS2
        d = rand(1:20)
        x = randn(d, 10)
        w = abs.(randn(size(x, 2))) .+ 0.1
        w_mat = abs.(randn(size(x))) .+ 0.1

        for a in [
                  sum_aggregation(Float64, d),
                  mean_aggregation(Float64, d),
                  max_aggregation(Float64, d),
                  lse_aggregation(Float64, d),
                  summeanmaxlse_aggregation(Float64, d),
                  SegmentedSum(randn(d)) |> Aggregation,
                  SegmentedMean(randn(d)) |> Aggregation,
                  SegmentedMax(randn(d)) |> Aggregation,
                  SegmentedLSE(randn(d), randn(d)) |> Aggregation,
                 ]
            @test gradtest(w -> a(x, bags, w), w)
            @test gradtest(w -> a(x, bags, w), w_mat)
        end

        # a = pnorm_aggregation(Float64, d)
        # @test_throws ErrorException gradtest(w -> a(x, bags, w), w)
        # @test_throws ErrorException gradtest(w -> a(x, bags, w), w_mat)
    end
end

@testset "aggregation grad check w.r.t. params" begin
    d = 5
    x = randn(d, 0)
    a1 = nonparam_aggregations(Float64, d)
    a2 = param_aggregations(Float64, d)
    @test gradtest(() -> a1(missing, ScatteredBags([Int[], Int[]])), Flux.params(a1))
    @test gradtest(() -> a1(missing, AlignedBags([0:-1]), nothing), Flux.params(a1))
    @test gradtest(() -> a1(x, ScatteredBags([Int[]])), Flux.params(a1))
    @test gradtest(() -> a1(x, AlignedBags([0:-1, 0:-1]), nothing), Flux.params(a1))
    @test gradtest(() -> a2(missing, ScatteredBags([Int[], Int[]])), Flux.params(a2))
    @test gradtest(() -> a2(missing, AlignedBags([0:-1]), nothing), Flux.params(a2))
    @test gradtest(() -> a2(x, ScatteredBags([Int[]])), Flux.params(a2))
    @test gradtest(() -> a2(x, AlignedBags([0:-1, 0:-1]), nothing), Flux.params(a2))

    for bags in BAGS2
        d = rand(1:20)
        x = randn(d, 10)
        a1 = nonparam_aggregations(Float64, d)
        a2 = param_aggregations(Float64, d)
        w = abs.(randn(size(x, 2))) .+ 0.1
        w_mat = abs.(randn(size(x))) .+ 0.1
        @test gradtest(() -> a1(x, bags), Flux.params(a1))
        @test gradtest(() -> a1(x, bags, w), Flux.params(a1))
        @test gradtest(() -> a1(x, bags, w_mat), Flux.params(a1))
        @test gradtest(() -> a2(x, bags), Flux.params(a2); atol=1e-3)
        @test gradtest(() -> a2(x, bags, w), Flux.params(a2); atol=1e-3)
        @test gradtest(() -> a2(x, bags, w_mat), Flux.params(a2); atol=1e-3)
    end
end

@testset "_typemin" begin
    @test Mill._typemin(Int) == typemin(Int)
    @test Mill._typemin(Missing) |> ismissing
    @test Mill._typemin(Union{Missing, Int}) == typemin(Int)
end
