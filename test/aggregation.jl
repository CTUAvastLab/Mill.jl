using Test, Mill, Flux
using Flux.Tracker: istracked
using Mill: segmented_pnorm, segmented_lse

let 
    X = Matrix{Float64}(reshape(1:12, 2, 6))
    BAGS = AlignedBags([1:1, 2:3, 4:6])
    W = [1, 1/2, 1/2, 1/8, 1/3, 13/24]
    C = [1, 2]

    @testset "basic aggregation functionality" begin
        @test SegmentedMean(2)(X, BAGS) ≈ [1.0 4.0 9.0; 2.0 5.0 10.0]
        @test SegmentedMax(2)(X, BAGS) ≈ [1.0 5.0 11.0; 2.0 6.0 12.0]
        @test SegmentedMeanMax(2)(X, BAGS) ≈ cat(SegmentedMean(2)(X, BAGS), SegmentedMax(2)(X, BAGS), dims=1)
        @test SegmentedMean(2)(X, BAGS, W) ≈ [1.0 4.0 236/24; 2.0 5.0 260/24]
        @test SegmentedMax(2)(X, BAGS, W) ≈ SegmentedMax(2)(X, BAGS)
        @test SegmentedMeanMax(2)(X, BAGS, W) ≈ cat(SegmentedMean(2)(X, BAGS, W), SegmentedMax(2)(X, BAGS, W), dims=1)

        @test typeof(SegmentedMean(randn(2))(X, BAGS)) <: Matrix
        @test typeof(SegmentedMax(rand(2))(X, BAGS)) <: Matrix
        @test typeof(SegmentedMean(randn(2))(Flux.param(X), BAGS)) <: Matrix
        @test typeof(SegmentedMax(rand(2))(Flux.param(X), BAGS)) <: Matrix
    end

    @testset "pnorm functionality" begin
        for t = 1:10
            a, b, c, d, p1, p2, c1, c2, w1, w2 = randn(10)
            w1 = abs(w1)
            w2 = abs(w2)
            bags = ScatteredBags([[1,2]])
            @test segmented_pnorm([a b; c d], C, [p1, p2], [c1, c2], bags) ≈ [
                    (1/2*(abs(a-c1)^p1 + abs(b-c1)^p1))^(1/p1);
                    (1/2*(abs(c-c2)^p2 + abs(d-c2)^p2))^(1/p2)
                ]
            @test segmented_pnorm([a b; c d], C, [p1, p2], [c1, c2], bags, [w1, w2]) ≈ [
                    (1/(w1+w2)*(w1*abs(a-c1)^p1 + w2*abs(b-c1)^p1))^(1/p1);
                    (1/(w1+w2)*(w1*abs(c-c2)^p2 + w2*abs(d-c2)^p2))^(1/p2)
                ]
            x = randn(2, 6)
            @test segmented_pnorm(x, C, [1, 1], [0, 0], BAGS) ≈ SegmentedMean(2)(abs.(x), BAGS)
            @test segmented_pnorm(x, C, [2, 2], [0, 0], BAGS) ≈ hcat([sqrt.(sum(x[:, b] .^ 2, dims=2) ./ length(b)) for b in BAGS]...)
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
            @test SegmentedMean(C)(missing, bags).data == repeat(C, 1, length(bags))
            @test SegmentedMax(C)(missing, bags).data == repeat(C, 1, length(bags))
            @test SegmentedLSE(dummy, C)(missing, bags).data == repeat(C, 1, length(bags))
            @test SegmentedPNorm(dummy, dummy, C)(missing, bags).data == repeat(C, 1, length(bags)) 
        end
    end

    @testset "testing stability with respect to tracker and non-tracked arrays" begin
            @test !istracked(SegmentedMean(C)(X, BAGS))
            @test !istracked(SegmentedMax(C)(X, BAGS))
            @test !istracked(SegmentedMean(C)(X, BAGS, W))
            @test !istracked(SegmentedMax(C)(X, BAGS, W))

            @test istracked(SegmentedMean(C)(param(X), BAGS))
            @test istracked(SegmentedMax(C)((param(X), BAGS))
            @test istracked(SegmentedMean(C)(X, BAGS, param(W)))
            @test istracked(SegmentedMax(C)(X, BAGS, param(W)))
            
            @test istracked(SegmentedMax(rand(2))(X, BAGS)) <: Matrix
            @test istracked(SegmentedMean(randn(2))(Flux.param(X), BAGS)) <: Matrix
            @test istracked(SegmentedMax(rand(2))(Flux.param(X), BAGS)) <: Matrix
        end

end
