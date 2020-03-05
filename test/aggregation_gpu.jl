# using Pkg
# Pkg.activate("..")
using CUDAnative, CuArrays, Mill, BenchmarkTools
using Mill: ∇dw_segmented_mean!, segmented_mean_forw, segmented_mean_back
using Mill: weight, bagnorm
using Flux
using Test
CuArrays.allowscalar(false)

@testset "segmented mean GPU" begin
	x = Matrix{Float64}(reshape(1:12, 2, 6))
	bags = AlignedBags([1:1, 2:3, 4:6, 0:-1])
	c = [1.0, -2]
	Δ = randn(2,4)
	gΔ = CuArray{Float32}(Δ)

	gx = CuArray{Float32}(x)
	gc = CuArray{Float32}(c)
	gbags = bags |> Mill.gpu
	for w in [nothing, abs.(randn(size(x,2))), abs.(randn(size(x)))]
		gw = isnothing(w) ? nothing : CuArray(w)
		y = segmented_mean_forw(x, c, bags, w)
		gy = segmented_mean_forw(gx, gc, gbags, gw)
		@test y ≈ gy

		Δx = segmented_mean_back(Δ, y, x, c, bags, w)[1]
		Δgx = segmented_mean_back(gΔ, gy, gx, gc, gbags, gw)[1]
		@test Δx ≈ Δgx

		Δc = segmented_mean_back(Δ, y, x, c, bags, w)[2]
		Δgc = segmented_mean_back(gΔ, gy, gx, gc, gbags, gw)[2]
		@test Δc ≈ Δgc

		Δw = segmented_mean_back(Δ, y, x, c, bags, w)[4]
		Δgw = segmented_mean_back(gΔ, gy, gx, gc, gbags, gw)[4]
		!isnothing(w) && @test Δw ≈ Δgw
	end
end

@testset "segmented max GPU" begin
	x = Matrix{Float64}(reshape(1:12, 2, 6))
	bags = AlignedBags([1:1, 2:3, 4:6, 0:-1])
	c = [1.0, -2]
	Δ = ones(2,4)
	gΔ = CuArray{Float32}(Δ)

	gx = CuArray{Float32}(x)
	gc = CuArray{Float32}(c)
	gbags = bags |> Mill.gpu

	y = Mill.segmented_max_forw(x, c, bags)
	gy, maxI = Mill.segmented_max_forw_maxI(gx, gc, gbags.bs, gbags.be)
	@test y ≈ gy

	Δx = Mill.segmented_max_back(Δ, y, x, c, bags)[1]
	Δgx = Mill.segmented_max_back(gΔ, maxI, gy, gx, gc, gbags)[1]
	@test Δx ≈ Δgx

	Δc = Mill.segmented_max_back(Δ, y, x, c, bags)[2]
	Δgc = Mill.segmented_max_back(gΔ, maxI, gy, gx, gc, gbags)[2]
	@test Δc ≈ Δgc
end


#Let's do a stress test
l = rand(1:100, 10000)
bags = Mill.length2bags(l);
x = randn(Float32, 160, sum(l));
c = randn(Float32, 160);
gx = CuArray(x)
gc = CuArray(c)
gbags = bags |> Mill.gpu

y = Mill.segmented_max_forw(x, c, bags)
gy, maxI = Mill.segmented_max_forw_maxI(gx, gc, gbags.bs, gbags.be)
y ≈ gy

Flux.gradient(gx -> sum(Mill.segmented_max_forw(gx, gc, gbags)), gx)

@btime Mill.segmented_max_forw(x, c, bags);
@btime CuArrays.@sync Mill.segmented_max_forw(gx, gc, gbags);
@btime Mill.segmented_mean_forw(x, c, bags, nothing);
@btime CuArrays.@sync Mill.segmented_mean_forw(gx, gc, gbags, nothing);
