using Flux.Tracker: istracked
import Mill: segmented_pnorm

let 
    X = Matrix{Float64}(reshape(1:12, 2, 6))
    d = size(X, 1)
    BAGS = [1:1, 2:3, 4:6]
    W = [1, 1/2, 1/2, 1/8, 1/3, 13/24]

    @testset "aggregation functionality" begin
        @test SegmentedMean()(X, BAGS) ≈ [1.0 4.0 9.0; 2.0 5.0 10.0]
        @test SegmentedMax()(X, BAGS) ≈ [1.0 5.0 11.0; 2.0 6.0 12.0]
        @test SegmentedMeanMax()(X, BAGS) ≈ cat(SegmentedMean()(X, BAGS), SegmentedMax()(X, BAGS), dims=1)
        @test SegmentedMean()(X, BAGS, W) ≈ [1.0 4.0 236/24; 2.0 5.0 260/24]
        @test SegmentedMax()(X, BAGS, W) ≈ SegmentedMax()(X, BAGS)
        @test SegmentedMeanMax()(X, BAGS, W) ≈ cat(SegmentedMean()(X, BAGS, W), SegmentedMax()(X, BAGS, W), dims=1)

        @test segmented_pnorm(X, [1, 1], [0, 0], BAGS) ≈ SegmentedMean()(abs.(X), BAGS)
        @test segmented_pnorm(X, [2, 2], [0, 0], BAGS) ≈ hcat([sqrt.(sum(X[:, b] .^ 2, dims=2) ./ length(b)) for b in BAGS]...)
    end

    @testset "right output aggregation types" begin
        Y = Flux.param(X)
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

        @test !istracked(PNorm(randn(d), randn(d))(X, BAGS))
        @test istracked(PNorm(randn(d), randn(d))(Y, BAGS))
        @test istracked(PNorm(d)(X, BAGS))
        @test istracked(PNorm(d)(Y, BAGS))

        @test !istracked(PNorm(randn(d), randn(d))(X, BAGS, W))
        @test istracked(PNorm(randn(d), randn(d))(Y, BAGS, W))
        @test istracked(PNorm(d)(X, BAGS, W))
        @test istracked(PNorm(d)(Y, BAGS, W))
    end
end
