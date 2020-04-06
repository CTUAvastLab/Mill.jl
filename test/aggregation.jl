using Test, Mill, Flux
using Mill: p_map, inv_p_map, r_map, inv_r_map
import Mill: bagnorm

@testset "basic aggregation functionality" begin
    W = [1, 1/2, 1/2, 1/8, 1/3, 13/24]
    X = Matrix{Float64}(reshape(1:12, 2, 6))
    bags = BAGS[1]
    @test SegmentedMean(2)(X, bags) ≈ [1.0 4.0 9.0; 2.0 5.0 10.0]
    @test SegmentedSum(2)(X, bags) ≈ length.(bags)' .* SegmentedMean(2)(X, bags)
    @test SegmentedMax(2)(X, bags) ≈ [1.0 5.0 11.0; 2.0 6.0 12.0]
    @test SegmentedMeanMax(2)(X, bags) ≈ cat(SegmentedMean(2)(X, bags), SegmentedMax(2)(X, bags), dims=1)
    @test SegmentedMean(2)(X, bags, W) ≈ [1.0 4.0 236/24; 2.0 5.0 260/24]
    @test SegmentedSum(2)(X, bags, W) ≈ [bagnorm(W, b) for b in bags]' .* SegmentedMean(2)(X, bags, W)
    @test SegmentedMax(2)(X, bags, W) ≈ SegmentedMax(2)(X, bags)
end

@testset "matrix weights" begin
    W = abs.(randn(6))
    W_mat = vcat(W, 2*W)
    X = Matrix{Float64}(reshape(1:12, 2, 6))
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
        agg = SegmentedPNorm([ρ1, ρ2], [c1, c2], dummy)
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
            agg = SegmentedPNorm(inv_p_map.([1+1e-16, 1+1e-16]), [0, 0], dummy)
            @test agg(X, bags) ≈ SegmentedMean(2)(abs.(X), bags)
            agg = SegmentedPNorm(inv_p_map.([2, 2]), [0, 0], dummy)
            @test agg(X, bags) ≈ hcat([sqrt.(sum(X[:, b] .^ 2, dims=2) ./ length(b)) for b in bags]...)
        end
    end
end

@testset "lse functionality" begin
    dummy = randn(2)
    for t = 1:10
        a, b, c, d, ρ1, ρ2 = randn(6)
        r1, r2 = r_map(ρ1), r_map(ρ2)
        @test SegmentedLSE([ρ1, ρ2], dummy)([a b; c d], ScatteredBags([[1,2]])) ≈ [
                                                                               1/r1*log(1/2*(exp(a*r1)+exp(b*r1)));
                                                                               1/r2*log(1/2*(exp(c*r2)+exp(d*r2)))
                                                                              ]
        for bags in BAGS
            X = randn(2, 6)
            W = abs.(randn(6))
            r1, r2 = randn(2)
            # doesn't use weights
            @test all(SegmentedLSE([ρ1, ρ2], dummy)(X, bags) .== SegmentedLSE([ρ1, ρ2], dummy)(X, bags, W))
            # the bigger value of r, the closer we are to the real maximum
            @test isapprox(SegmentedLSE([100, 100], dummy)(X, bags), SegmentedMax(2)(X, bags), atol=0.1)
        end
    end
end

@testset "lse numerical stability" begin
    # it holds for any c and any r != 0, LSE(r)(x) == c .+ LSE(r)(x .- c)
    b1 = AlignedBags([1:2])
    b2 = ScatteredBags([[1,2]])
    dummy = randn(2)
    for t in 1:5
        ρ = randn(2)
        @test SegmentedLSE(ρ, dummy)([1e15 1e15; 1e15 1e15], b1) ≈ [1e15; 1e15]
        @test SegmentedLSE(ρ, dummy)([-1e15 -1e15; -1e15 -1e15], b2) ≈ [-1e15; -1e15]
        @test SegmentedLSE(ρ, dummy)([1e15 1e15; 1e15 1e15], b1, [1, 1]) ≈ [1e15; 1e15]
        @test SegmentedLSE(ρ, dummy)([1e15 1e15; 1e15 1e15], b2, [2, 2]) ≈ [1e15; 1e15]
        @test SegmentedLSE(ρ, dummy)([-1e15 -1e15; -1e15 -1e15], b1, [1, 1]) ≈ [-1e15; -1e15]
        @test SegmentedLSE(ρ, dummy)([-1e15 -1e15; -1e15 -1e15], b2, [2, 2]) ≈ [-1e15; -1e15]
    end
end

@testset "missing values" begin
    dummy = randn(2)
    C = randn(2)
    for bags in [AlignedBags([0:-1]), AlignedBags([0:-1, 0:-1, 0:-1])]
        @test SegmentedMean(C)(missing, bags) == repeat(C, 1, length(bags))
        @test SegmentedSum(C)(missing, bags) == repeat(C, 1, length(bags))
        @test SegmentedMax(C)(missing, bags) == repeat(C, 1, length(bags))
        @test SegmentedLSE(dummy, C)(missing, bags) == repeat(C, 1, length(bags))
        @test SegmentedPNorm(dummy, dummy, C)(missing, bags) == repeat(C, 1, length(bags)) 
    end

    # default values C are indeed filled in
    for bags in vcat(BAGS2)
        idcs = isempty.(bags.bags)
        l = maximum(maximum.(bags.bags[.!idcs]))
        X = randn(2, l)
        @test SegmentedMean(C)(X, bags)[:, idcs] == repeat(C, 1, sum(idcs))
        @test SegmentedSum(C)(X, bags)[:, idcs] == repeat(C, 1, sum(idcs))
        @test SegmentedMax(C)(X, bags)[:, idcs] == repeat(C, 1, sum(idcs))
        @test SegmentedLSE(dummy, C)(X, bags)[:, idcs] == repeat(C, 1, sum(idcs))
        @test SegmentedPNorm(dummy, dummy, C)(X, bags)[:, idcs] == repeat(C, 1, sum(idcs))
    end
end
