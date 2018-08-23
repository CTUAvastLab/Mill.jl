using Flux
import Mill: segmented_mean, segmented_max, segmented_meanmax, segmented_weighted_mean, segmented_weighted_max, segmented_weighted_meanmax, segmented_pnorm, p_map, inv_p_map, PNorm
import Mill: BagModel, ChainModel
import Flux.Tracker: gradcheck

x1 = randn(2,10)
x2 = Matrix{Float64}(reshape(1:12, 2, 6))
bags1 = [1:5,6:10]
bags2 = [1:1, 2:3, 4:6]
w = [1, 1/2, 1/2, 1/8, 1/3, 13/24]

@testset "aggregation grad check" begin
	@test gradcheck(x -> sum(segmented_mean(x, [1:4])), randn(4, 4))
	@test gradcheck(x -> sum(segmented_mean(x, [1:2, 3:4, 5:5, 6:8])), randn(4, 8))
	@test gradcheck(x -> sum(segmented_mean(x, [1:2, 3:4, 0:-1, 5:5, 6:8])), randn(4, 8))
	@test gradcheck(x -> sum(segmented_max(x, [1:4])), randn(4, 4))
	@test gradcheck(x -> sum(segmented_max(x, [1:4, 5:5, 6:8])), randn(4, 8))
	@test gradcheck(x -> sum(segmented_meanmax(x, [1:4])), randn(4, 4))
	@test gradcheck(x -> sum(segmented_meanmax(x, [1:4, 5:5, 6:8])), randn(4, 8))

	for g in [	x -> sum(segmented_mean(x, bags1)),
				x -> sum(segmented_max(x, bags1)),
				x -> sum(segmented_meanmax(x, bags1)),
		]
		@test gradcheck(g, x1)
	end
	for g in [  x -> sum(segmented_weighted_mean(x, bags2, w)),
				x -> sum(segmented_weighted_max(x, bags2, w)),
				x -> sum(segmented_weighted_meanmax(x, bags2, w))
		]
		@test gradcheck(g, x2)
	end

	for n = 1:10
		A = 100 .* randn() .* randn(2, 10)
		B = randn(2, 6)
		pn = PNorm(2)
		@test gradcheck(x -> sum(segmented_pnorm(x, bags1, pn.ρ, pn.c)), A)
		@test gradcheck(ρ -> sum(segmented_pnorm(A, bags1, ρ, pn.c)), pn.ρ)
		@test gradcheck(c -> sum(segmented_pnorm(A, bags1, pn.ρ, c)), pn.c)
		@test gradcheck(x -> sum(segmented_pnorm(x, bags2, pn.ρ, pn.c)), B)
		@test gradcheck(ρ -> sum(segmented_pnorm(B, bags2, ρ, pn.c)), pn.ρ)
		@test gradcheck(c -> sum(segmented_pnorm(B, bags2, pn.ρ, c)), pn.c)
	end
end

@testset "aggregation functionality" begin
	@test segmented_mean(x2, bags2) ≈ [1.0 4.0 9.0; 2.0 5.0 10.0]
	@test segmented_max(x2, bags2) ≈ [1.0 5.0 11.0; 2.0 6.0 12.0]
	@test segmented_meanmax(x2, bags2) ≈ cat(1, segmented_max(x2, bags2), segmented_mean(x2, bags2))
	@test segmented_weighted_mean(x2, bags2, w) ≈ [1.0 4.0 236/24; 2.0 5.0 260/24]
	@test segmented_weighted_max(x2, bags2, w) ≈ segmented_max(x2, bags2)
	@test segmented_weighted_meanmax(x2, bags2, w) ≈ cat(1, segmented_weighted_max(x2, bags2, w), segmented_weighted_mean(x2, bags2, w))

	@test segmented_pnorm(x2, bags2, inv_p_map.([1, 1]), [0, 0]) ≈ segmented_mean(abs.(x2), bags2)
	@test segmented_pnorm(x2, bags2, inv_p_map.([2, 2]), [0, 0]) ≈ hcat([sqrt.(sum(x2[:, b] .^ 2, 2) ./ length(b)) for b in bags2]...)
end

# @testset "learning basic pnorm types" begin
# 	model = PNorm(1)
# 	loss(x,y) = Flux.logitcrossentropy(model(x), Flux.onehotbatch(y, 1:2));
#
# 	data, y =
# 	dataset = RandomBatches((data,y), 100, 2000)
# 	evalcb = () -> @show(loss(data, y))
# 	opt = Flux.ADAM(params(model))
# 	Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))
#
# end
