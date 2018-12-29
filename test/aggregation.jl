using Flux.Tracker: istracked
import Mill: _segmented_pnorm, _segmented_lse, PNorm, LSE

let 
    X = Matrix{Float64}(reshape(1:12, 2, 6))
    BAGS = AlignedBags([1:1, 2:3, 4:6])
    W = [1, 1/2, 1/2, 1/8, 1/3, 13/24]

    @testset "aggregation functionality" begin
        @test SegmentedMean()(X, BAGS) ≈ [1.0 4.0 9.0; 2.0 5.0 10.0]
        @test SegmentedMax()(X, BAGS) ≈ [1.0 5.0 11.0; 2.0 6.0 12.0]
        @test SegmentedMeanMax()(X, BAGS) ≈ cat(SegmentedMean()(X, BAGS), SegmentedMax()(X, BAGS), dims=1)
        @test SegmentedMean()(X, BAGS, W) ≈ [1.0 4.0 236/24; 2.0 5.0 260/24]
        @test SegmentedMax()(X, BAGS, W) ≈ SegmentedMax()(X, BAGS)
        @test SegmentedMeanMax()(X, BAGS, W) ≈ cat(SegmentedMean()(X, BAGS, W), SegmentedMax()(X, BAGS, W), dims=1)

        for t = 1:10
            a, b, c, d, p1, p2, c1, c2, w1, w2 = randn(10)
            w1 = abs(w1)
            w2 = abs(w2)
            @test _segmented_pnorm([a b; c d], [p1, p2], [c1, c2], ScatteredBags([[1,2]])) ≈ [
                    (1/2*(abs(a-c1)^p1 + abs(b-c1)^p1))^(1/p1);
                    (1/2*(abs(c-c2)^p2 + abs(d-c2)^p2))^(1/p2)
                ]
            @test _segmented_pnorm([a b; c d], [p1, p2], [c1, c2], ScatteredBags([[1,2]]), [w1, w2]) ≈ [
                    (1/(w1+w2)*(w1*abs(a-c1)^p1 + w2*abs(b-c1)^p1))^(1/p1);
                    (1/(w1+w2)*(w1*abs(c-c2)^p2 + w2*abs(d-c2)^p2))^(1/p2)
                ]
            x = randn(2, 6)
            @test _segmented_pnorm(x, [1, 1], [0, 0], BAGS) ≈ SegmentedMean()(abs.(x), BAGS)
            @test _segmented_pnorm(x, [2, 2], [0, 0], BAGS) ≈ hcat([sqrt.(sum(x[:, b] .^ 2, dims=2) ./ length(b)) for b in BAGS]...)
        end

        for t = 1:10
            a, b, c, d, r1, r2 = randn(6)
            @test LSE([r1, r2])([a b; c d], ScatteredBags([[1,2]])) ≈ [
                    1/r1*log(1/2*(exp(a*r1)+exp(b*r1)));
                    1/r2*log(1/2*(exp(c*r2)+exp(d*r2)))
                ]
            X = randn(2, 6)
            r1, r2 = randn(2)
            @test all(LSE([r1, r2])(X, BAGS) .== LSE([r1, r2])(X, BAGS, W))
            # the bigger value of r, the closer we are to the real maximum
            @test isapprox(LSE([100, 100])(X, BAGS), SegmentedMax()(X, BAGS), atol=0.1)
        end
    end

    @testset "lse numerical stability" begin
        b1 = AlignedBags([1:2])
        b2 = ScatteredBags([[1,2]])
        @test LSE([1,1])([1e15 1e15; 1e15 1e15], b1) ≈ [1e15; 1e15]
        @test LSE([1,1])([-1e15 -1e15; -1e15 -1e15], b2) ≈ [-1e15; -1e15]
        @test LSE([1,1])([1e15 1e15; 1e15 1e15], b1, [1, 1]) ≈ [1e15; 1e15]
        @test LSE([1,1])([1e15 1e15; 1e15 1e15], b2, [2, 2]) ≈ [1e15; 1e15]
        @test LSE([1,1])([-1e15 -1e15; -1e15 -1e15], b1, [1, 1]) ≈ [-1e15; -1e15]
        @test LSE([1,1])([-1e15 -1e15; -1e15 -1e15], b2, [2, 2]) ≈ [-1e15; -1e15]
    end

    @testset "right output aggregation types" begin
        Y = Flux.param(X)
        dim = size(X, 1)
        @test !istracked(SegmentedMean()(X, BAGS))
        @test !istracked(SegmentedMax()(X, BAGS))
        @test !istracked(SegmentedMeanMax()(X, BAGS))
        @test !istracked(SegmentedMean()(X, BAGS, W))
        @test !istracked(SegmentedMax()(X, BAGS, W))
        @test !istracked(SegmentedMeanMax()(X, BAGS, W))

        @test istracked(SegmentedMean()(Y, BAGS))
        @test istracked(SegmentedMax()(Y, BAGS))
        @test istracked(SegmentedMeanMax()(Y, BAGS))
        @test istracked(SegmentedMean()(Y, BAGS, W))
        @test istracked(SegmentedMax()(Y, BAGS, W))
        @test istracked(SegmentedMeanMax()(Y, BAGS, W))

        @test !istracked(PNorm(randn(dim), randn(dim))(X, BAGS))
        @test istracked(PNorm(randn(dim), randn(dim))(Y, BAGS))
        @test istracked(PNorm(dim)(X, BAGS))
        @test istracked(PNorm(dim)(Y, BAGS))

        @test !istracked(PNorm(randn(dim), randn(dim))(X, BAGS, W))
        @test istracked(PNorm(randn(dim), randn(dim))(Y, BAGS, W))
        @test istracked(PNorm(dim)(X, BAGS, W))
        @test istracked(PNorm(dim)(Y, BAGS, W))

        @test !istracked(LSE(randn(dim))(X, BAGS))
        @test istracked(LSE(randn(dim))(Y, BAGS))
        @test istracked(LSE(dim)(X, BAGS))
        @test istracked(LSE(dim)(Y, BAGS))

        @test !istracked(LSE(randn(dim))(X, BAGS, W))
        @test istracked(LSE(randn(dim))(Y, BAGS, W))
        @test istracked(LSE(dim)(X, BAGS, W))
        @test istracked(LSE(dim)(Y, BAGS, W))
    end
end
