using Test, Flux, SparseArrays, Mill
using Mill: BagConv, convsum, bagconv, legacy_bagconv, _convshift, ∇convsum, ArrayNode, BagNode, ∇wbagconv, ∇xbagconv

@testset "testing convolution shift" begin
    @test _convshift(2) == 0:1
    @test _convshift(3) == -1:1
    @test _convshift(4) == -1:2
    @test _convshift(5) == -2:2
end

@testset "testing matvec and vecvec products " begin
    W = randn(3, 4)
    Wt = Matrix(transpose(W))
    xs = sprand(4, 10, 0.5)
    x = Matrix(xs)

    o = zeros(3, 10)
    foreach(i -> Mill._addmatvec!(o, i, W, x, i), 1:10)
    @test o ≈ W*x
    fill!(o, 0)
    foreach(i -> Mill._addmatvec!(o, i, W, xs, i), 1:10)
    @test o ≈ W*x

    o = zeros(3, 1)
    foreach(i -> Mill._addmatvec!(o, 1, W, x, i), 1:10)
    @test o ≈ sum(W*x, dims = 2)
    fill!(o, 0)
    foreach(i -> Mill._addmatvec!(o, 1, W, xs, i), 1:10)
    @test o ≈ sum(W*x, dims = 2)

    o = zeros(3, 10)
    foreach(i -> Mill._addmattvec!(o, i, Wt, x, i), 1:10)
    @test o ≈ W*x

    o = zeros(3, 1)
    foreach(i -> Mill._addmattvec!(o, 1, Wt, x, i), 1:10)
    @test o ≈ sum(W*x, dims = 2)

    xs = sprand(10, 1, 0.5)
    r, s = randn(10,1), Matrix(xs)
    o = zeros(10, 10)
    Mill._addvecvect!(o, r, 1, s, 1)
    @test o ≈ r * transpose(s)
    fill!(o, 0)
    Mill._addvecvect!(o, r, 1, xs, 1)
    @test o ≈ r * transpose(s)
end

@testset "testing forward convolution & gradient" begin
    x = Float64.([1 10  100  1000  10000]);
    y = 2 .* x;
    z = 4 .* x;
    Δ = ones(1, 5)
    for bags in [AlignedBags([1:2,3:5]), ScatteredBags([[1,2],[3,4,5]])];

        @test convsum(bags, x) == x
        @test convsum(bags, x, y) == [21  10  2100  21000  10000]
        @test convsum(bags, x, y, z) == [42  21  4200  42100  21000]

        @test all(ngradient(x -> sum(convsum(bags,x, y, z)), x)[1] .== ∇convsum(Δ, bags, 3)[1])
        @test all(ngradient(y -> sum(convsum(bags,x, y, z)), y)[1] .== ∇convsum(Δ, bags, 3)[2])
        @test all(ngradient(z -> sum(convsum(bags,x, y, z)), z)[1] .== ∇convsum(Δ, bags, 3)[3])

        @test gradtest((a, b, c) -> convsum(bags, a, b, c), x, y, z)
    end
end

@testset "testing the convolution" begin
    xs = sprand(3, 15, 0.5)
    x = Matrix(xs)
    filters = randn(4, 3, 3)
    fs = [filters[:,:,i] for i in 1:3]
    for bags in [AlignedBags([1:1, 2:3, 4:6, 7:15]), ScatteredBags(collect.([1:1, 2:3, 4:6, 7:15]))]
        @test bagconv(x, bags, fs...) ≈ legacy_bagconv(x, bags, filters)
        @test bagconv(x, bags, fs...) ≈ bagconv(xs, bags, fs...)
        @test isapprox(ngradient(f -> sum(bagconv(x, bags, f, fs[2], fs[3])), fs[1])[1],  ∇wbagconv(ones(4, 15), x, bags, fs...)[1], rtol = 1e-6)
        @test isapprox(ngradient(f -> sum(bagconv(xs, bags, f, fs[2], fs[3])), fs[1])[1],  ∇wbagconv(ones(4, 15), xs, bags, fs...)[1], rtol = 1e-6)
        @test isapprox(ngradient(x -> sum(bagconv(x, bags, fs...)), x)[1],  ∇xbagconv(ones(4, 15), x, bags, fs...), rtol = 1e-5)
        @test gradtest((a, b, c) -> bagconv(x, bags, a, b, c), fs...)
        @test gradtest((a, b, c) -> bagconv(xs, bags, a, b, c), fs...)
        @test gradtest(x -> bagconv(x, bags, fs...), x)
        @test gradtest((x, a, b , c) -> bagconv(x, bags, a, b, c), x, fs...)
    end
end

@testset "testing convolution with ScatteredBags" begin
    xs = sprand(3, 7, 0.5)
    x = Matrix(xs)
    filters = randn(4, 3, 3)
    fs = [filters[:,:,i] for i in 1:3]
    baga = AlignedBags([1:3, 4:7])
    bags = ScatteredBags([[1,2,3],[4,5,6,7]])
    bagsp = ScatteredBags([[1,4,7],[2,3,5,6]])
    xp = x[:, [1,4,5,2,6,7,3]]

    @testset "Test that all versions of convolutions are equal" begin
        @test bagconv(x, baga, fs...) == bagconv(x, bags, fs...)
        @test x[:,bags[1]] == xp[:,bagsp[1]]
        @test x[:,bags[2]] == xp[:,bagsp[2]]
        @test bagconv(x, bags, fs...)[:,bags[1]] == bagconv(xp, bagsp, fs...)[:,bagsp[1]]
        @test bagconv(x, bags, fs...)[:,bags[2]] == bagconv(xp, bagsp, fs...)[:,bagsp[2]]
    end

    @testset "Test that gradient of scattered convolution is correct" begin
        @test isapprox(ngradient(f -> sum(bagconv(xp, bagsp, f, fs[2], fs[3])), fs[1])[1],  ∇wbagconv(ones(4, 15), xp, bagsp, fs...)[1], rtol = 1e-6)
        @test isapprox(ngradient(f -> sum(bagconv(x, bags, f, fs[2], fs[3])), fs[1])[1],  ∇wbagconv(ones(4, 15), xp, bagsp, fs...)[1], rtol = 1e-6)
        @test isapprox(ngradient(x -> sum(bagconv(x, bagsp, fs...)), xp)[1],  ∇xbagconv(ones(4, 15), xp, bagsp, fs...), rtol = 1e-5)
        @test gradtest((a, b, c) -> bagconv(x, bags, a, b, c), fs...)
        @test gradtest(x -> bagconv(x, bagsp, fs...), xp)
        @test gradtest((x, a, b , c) -> bagconv(x, bagsp, a, b, c), xp, fs...)
    end
end

@testset "testing convolution layer" begin
    xs = sprand(3, 15, 0.5)
    x = Matrix(xs)
    for bags in [AlignedBags([1:1, 2:3, 4:6, 7:15]), ScatteredBags(collect.([1:1, 2:3, 4:6, 7:15]))]
        ds = BagNode(ArrayNode(Float32.(x)), bags)

        m = BagConv(3, 4, 3, relu)
        @test length(params(m)) == 3
        @test size(m(x, bags)) == (4, 15)
        @test size(m(xs, bags)) == (4, 15)

        m = BagConv(3, 4, 1)
        @test size(m(x, bags)) == (4, 15)
        @test size(m(xs, bags)) == (4, 15)

        m = BagChain(BagConv(3, 4, 3, relu), BagConv(3, 4, 2))
        @test length(params(m)) == 5
    end
end
