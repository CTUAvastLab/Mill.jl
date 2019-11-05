using Test, Mill, Flux
using Mill: p_map, inv_p_map

X = Matrix{Float64}(reshape(1:12, 2, 6))
BAGS = AlignedBags([1:1, 2:3, 4:6])
W = [1, 1/2, 1/2, 1/8, 1/3, 13/24]
W_mat = vcat(W, 2*W)
C = [1, 2]

@testset "basic aggregation functionality" begin
    @test SegmentedMean(2)(X, BAGS) ≈ [1.0 4.0 9.0; 2.0 5.0 10.0]
    @test SegmentedMax(2)(X, BAGS) ≈ [1.0 5.0 11.0; 2.0 6.0 12.0]
    @test SegmentedMeanMax(2)(X, BAGS) ≈ cat(SegmentedMean(2)(X, BAGS), SegmentedMax(2)(X, BAGS), dims=1)
    @test SegmentedMean(2)(X, BAGS, W) ≈ [1.0 4.0 236/24; 2.0 5.0 260/24]
    @test SegmentedMax(2)(X, BAGS, W) ≈ SegmentedMax(2)(X, BAGS)
    @test SegmentedMeanMax(2)(X, BAGS, W) ≈ cat(SegmentedMean(2)(X, BAGS, W), SegmentedMax(2)(X, BAGS, W), dims=1)
end

@testset "matrix functionality" begin
    @test SegmentedMean(2)(X, BAGS, W_mat) ≈ SegmentedMean(2)(X, BAGS, W)
end

@testset "pnorm functionality" begin
    for t = 1:10
        a, b, c, d, ρ1, ρ2, c1, c2, w1, w2 = randn(10)
        p1, p2 = p_map(ρ1), p_map(ρ2)
        w1 = abs(w1)
        w2 = abs(w2)
        bags = ScatteredBags([[1,2]])
        agg = SegmentedPNorm([ρ1, ρ2], [c1, c2], C)
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
        x = randn(2, 6)
        agg = SegmentedPNorm(inv_p_map.([1+1e-16, 1+1e-16]), [0, 0], C)
        @test agg(x, BAGS) ≈ SegmentedMean(2)(abs.(x), BAGS)
        agg = SegmentedPNorm(inv_p_map.([2, 2]), [0, 0], C)
        @test agg(x, BAGS) ≈ hcat([sqrt.(sum(x[:, b] .^ 2, dims=2) ./ length(b)) for b in BAGS]...)
    end
end

@testset "lse functionality" begin
    for t = 1:10
        a, b, c, d, r1, r2 = randn(6)
        @test SegmentedLSE([r1, r2], C)([a b; c d], ScatteredBags([[1,2]])) ≈ [
                                                                               1/r1*log(1/2*(exp(a*r1)+exp(b*r1)));
                                                                               1/r2*log(1/2*(exp(c*r2)+exp(d*r2)))
                                                                              ]
        X = randn(2, 6)
        r1, r2 = randn(2)
        @test all(SegmentedLSE([r1, r2], C)(X, BAGS) .== SegmentedLSE([r1, r2], C)(X, BAGS, W))
        # the bigger value of r, the closer we are to the real maximum
        @test isapprox(SegmentedLSE([100, 100], C)(X, BAGS), SegmentedMax(2)(X, BAGS), atol=0.1)
    end
end

@testset "lse numerical stability" begin
    b1 = AlignedBags([1:2])
    b2 = ScatteredBags([[1,2]])
    @test SegmentedLSE([1,1], C)([1e15 1e15; 1e15 1e15], b1) ≈ [1e15; 1e15]
    @test SegmentedLSE([1,1], C)([-1e15 -1e15; -1e15 -1e15], b2) ≈ [-1e15; -1e15]
    @test SegmentedLSE([1,1], C)([1e15 1e15; 1e15 1e15], b1, [1, 1]) ≈ [1e15; 1e15]
    @test SegmentedLSE([1,1], C)([1e15 1e15; 1e15 1e15], b2, [2, 2]) ≈ [1e15; 1e15]
    @test SegmentedLSE([1,1], C)([-1e15 -1e15; -1e15 -1e15], b1, [1, 1]) ≈ [-1e15; -1e15]
    @test SegmentedLSE([1,1], C)([-1e15 -1e15; -1e15 -1e15], b2, [2, 2]) ≈ [-1e15; -1e15]
end


@testset "missing values" begin
    dummy = randn(2)
    for bags in [AlignedBags([0:-1]), AlignedBags([0:-1, 0:-1, 0:-1])]
        @test SegmentedMean(C)(missing, bags) == repeat(C, 1, length(bags))
        @test SegmentedMax(C)(missing, bags) == repeat(C, 1, length(bags))
        @test SegmentedLSE(dummy, C)(missing, bags) == repeat(C, 1, length(bags))
        @test SegmentedPNorm(dummy, dummy, C)(missing, bags) == repeat(C, 1, length(bags)) 
    end
end
