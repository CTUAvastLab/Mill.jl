using Test, Flux
using Mill: convsum, convmil, _convshift, ∇convsum
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

@testset "testing the convolution" begin
	x = randn(3, 15)
	bags = [1:1, 2:3, 4:6, 7:15]
	filters = randn(4, 3, 3)

	@test gradtest(x -> convmil(x, bags, filters), x)
	@test gradtest(ff -> convmil(x, bags, ff), filters)
end
