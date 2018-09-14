import Mill: segmented_pnorm
using Flux.Tracker: istracked

X = Matrix{Float64}(reshape(1:12, 2, 6))
d = size(X, 1)
BAGS = [1:1, 2:3, 4:6]
W = [1, 1/2, 1/2, 1/8, 1/3, 13/24]

@testset "aggregation functionality" begin
    @test segmented_mean(X, BAGS) ≈ [1.0 4.0 9.0; 2.0 5.0 10.0]
    @test segmented_max(X, BAGS) ≈ [1.0 5.0 11.0; 2.0 6.0 12.0]
    @test segmented_meanmax(X, BAGS) ≈ cat(segmented_mean(X, BAGS), segmented_max(X, BAGS), dims=1)
    @test segmented_mean(X, BAGS, W) ≈ [1.0 4.0 236/24; 2.0 5.0 260/24]
    @test segmented_max(X, BAGS, W) ≈ segmented_max(X, BAGS)
    @test segmented_meanmax(X, BAGS, W) ≈ cat(segmented_mean(X, BAGS, W), segmented_max(X, BAGS, W), dims=1)

    @test segmented_pnorm(X, [1, 1], [0, 0], BAGS) ≈ segmented_mean(abs.(X), BAGS)
    @test segmented_pnorm(X, [2, 2], [0, 0], BAGS) ≈ hcat([sqrt.(sum(X[:, b] .^ 2, 2) ./ length(b)) for b in BAGS]...)
end

@testset "right output aggregation types" begin
    Y = Flux.param(X)
    @test !istracked(segmented_mean(X, BAGS))
    @test !istracked(segmented_max(X, BAGS))
    @test !istracked(segmented_meanmax(X, BAGS))
    @test !istracked(segmented_mean(X, BAGS, W))
    @test !istracked(segmented_max(X, BAGS, W))
    @test !istracked(segmented_meanmax(X, BAGS, W))

    @test istracked(segmented_mean(Y, BAGS))
    @test istracked(segmented_max(Y, BAGS))
    @test istracked(segmented_meanmax(Y, BAGS))
    @test istracked(segmented_mean(Y, BAGS, W))
    @test istracked(segmented_max(Y, BAGS, W))
    @test istracked(segmented_meanmax(Y, BAGS, W))

    @test !istracked(PNorm(randn(d), randn(d))(X, BAGS))
    @test istracked(PNorm(randn(d), randn(d))(Y, BAGS))
    @test istracked(PNorm(d)(X, BAGS))
    @test istracked(PNorm(d)(Y, BAGS))

    @test !istracked(PNorm(randn(d), randn(d))(X, BAGS, W))
    @test istracked(PNorm(randn(d), randn(d))(Y, BAGS, W))
    @test istracked(PNorm(d)(X, BAGS, W))
    @test istracked(PNorm(d)(Y, BAGS, W))
end
