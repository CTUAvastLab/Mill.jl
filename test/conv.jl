using Test, Flux, SparseArrays, Mill
using Mill: convsum, bagconv, legacy_bagconv, _convshift, ∇convsum, ArrayNode, BagNode, ∇wbagconv
using Flux.Tracker: TrackedReal, gradcheck, grad, checkpoint
gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)

@testset "testing convolution shift" begin
	@test _convshift(2) == 0:1
	@test _convshift(3) == -1:1
	@test _convshift(4) == -1:2
	@test _convshift(5) == -2:2
end

@testset "testing forward convolution & gradient" begin
	x = Float64.([1 10  100  1000  10000]);
	y = 2 .* x;
	z = 4 .* x;
	Δ = ones(1, 5)
	bags = [1:2,3:5];

	@test convsum(bags, x) == x
	@test convsum(bags, x, y) == [21  10  2100  21000  10000]
	@test convsum(bags, x, y, z) == [42  21  4200  42100  21000]

	@test all(Flux.Tracker.ngradient(x -> sum(convsum(bags,x, y, z)), x)[1] .== ∇convsum(Δ, bags, 3)[1])
	@test all(Flux.Tracker.ngradient(y -> sum(convsum(bags,x, y, z)), y)[1] .== ∇convsum(Δ, bags, 3)[2])
	@test all(Flux.Tracker.ngradient(z -> sum(convsum(bags,x, y, z)), z)[1] .== ∇convsum(Δ, bags, 3)[3])

	@test gradtest((a, b, c) -> convsum(bags, a, b, c), x, y, z)
end

@testset "testing matvec and vecvec products " begin 
	W = randn(3, 4)
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

	xs = sprand(10, 1, 0.5)
	r, s = randn(10,1), Matrix(xs)
	o = zeros(10, 10)
	Mill._addvecvect!(o, r, 1, s, 1)
	@test o ≈ r * transpose(s)
	fill!(o, 0)
	Mill._addvecvect!(o, r, 1, xs, 1)
	@test o ≈ r * transpose(s)
end

@testset "testing the convolution" begin
	xs = sprand(3, 15, 0.5)
	x = Matrix(xs)
	bags = [1:1, 2:3, 4:6, 7:15]
	filters = randn(4, 3, 3)
	fs = [filters[:,:,i] for i in 1:3]

	@test bagconv(x, bags, fs...) ≈ legacy_bagconv(x, bags, filters)
	@test bagconv(x, bags, fs...) ≈ bagconv(xs, bags, fs...)
	@test isapprox(Flux.Tracker.ngradient(f -> sum(bagconv(x, bags, f, fs[2], fs[3])), fs[1])[1],  ∇wbagconv(ones(4, 15), x, bags, fs...)[1], atol = 1e-6)
	@test isapprox(Flux.Tracker.ngradient(f -> sum(bagconv(xs, bags, f, fs[2], fs[3])), fs[1])[1],  ∇wbagconv(ones(4, 15), xs, bags, fs...)[1], atol = 1e-6)
	@test gradtest((a, b, c) -> bagconv(x, bags, a, b, c), fs...)
	@test gradtest((a, b, c) -> bagconv(xs, bags, a, b, c), fs...)
end


# x = BagNode(ArrayNode(rand(3,10)),[1:3,4:10])

# k = 7;

# im = ArrayModel(Dense(size(x.data.data, 1), k, relu))
# tmpx = im(x.data)
# am = Mill.BagChain(BagConv(size(tmpx.data, 1), k, 3), Mill.SegmentedMeanMax())
# tmpx = am(xx, x.bags)
# bm = ArrayModel(Dense(size(tmpx.data, 1), 2))
# m = BagModel(im, am, bm)

# #let's test the sequential mapping of sites
# using Mill: ArrayNode, BagNode, SequentialBagNode, nobs, catobs

# @testset "Sequential bag node" begin
# 	a = SequentialBagNode(ArrayNode(rand(3,4)),[1:4], ["metadata", "metadata", "metadata", "metadata"])
#   b = SequentialBagNode(ArrayNode(rand(3,4)),[1:2,3:4])
#   c = SequentialBagNode(ArrayNode(rand(3,4)),[1:1,2:2,3:4], ["metadata", "metadata", "metadata", "metadata"])
#   d = SequentialBagNode(ArrayNode(rand(3,4)),[1:4,0:-1])

# 	@testset "testing nobs" begin
#     @test nobs(a) == 1
#     @test nobs(b) == 2
#     @test nobs(c) == 3
#     @test nobs(d) == 2
#   end

#   @testset "testing BagNode hcat" begin
#       @test all(catobs(a, b, c).data.data .== hcat(a.data.data, b.data.data, c.data.data))
#       @test all(reduce(catobs, [a, b, c]).data.data .== hcat(a.data.data, b.data.data, c.data.data))
#       @test all(catobs(a, b, c).bags .== [1:4, 5:6, 7:8, 9:9, 10:10, 11:12])
#       @test all(reduce(catobs, [a, b, c]).bags .== [1:4, 5:6, 7:8, 9:9, 10:10, 11:12])
#       @test all(catobs(c, a).data.data .== hcat(c.data.data, a.data.data))
#       @test all(reduce(catobs, [c, a]).data.data .== hcat(c.data.data, a.data.data))
#       @test all(catobs(c, a).bags .== [1:1, 2:2, 3:4, 5:8])
#       @test all(reduce(catobs, [c, a]).bags .== [1:1, 2:2, 3:4, 5:8])
#       @test all(catobs(a, c).data.data .== hcat(a.data.data, c.data.data))
#       @test all(reduce(catobs, [a, c]).data.data .== hcat(a.data.data, c.data.data))
#       @test all(catobs(a, c).bags .== [1:4, 5:5, 6:6, 7:8])
#       @test all(reduce(catobs, [a, c]).bags .== [1:4, 5:5, 6:6, 7:8])
#       @test all(catobs(a, d).data.data .== hcat(a.data.data, d.data.data))
#       @test all(reduce(catobs, [a, d]).data.data .== hcat(a.data.data, d.data.data))
#       @test all(catobs(a, d).bags .== [1:4, 5:8, 0:-1])
#       @test all(reduce(catobs, [a, d]).bags .== [1:4, 5:8, 0:-1])
#       @test all(catobs(d, a).data.data .== hcat(d.data.data, a.data.data))
#       @test all(reduce(catobs, [d, a]).data.data .== hcat(d.data.data, a.data.data))
#       @test all(catobs(d, a).bags .== [1:4, 0:-1, 5:8])
#       @test all(reduce(catobs, [d, a]).bags .== [1:4, 0:-1, 5:8])
#       @test all(catobs(d).data.data .== hcat(d.data.data))
#       @test all(reduce(catobs, [d]).data.data .== hcat(d.data.data))
#       @test all(catobs(d).bags .== [1:4, 0:-1])
#       @test all(reduce(catobs, [d]).bags .== [1:4, 0:-1])
#   end
# end
