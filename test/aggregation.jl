using Flux
import Mill: segmented_mean
import Mill: segmented_max
import Mill: PNorm
import Mill: segmented_meanmax
import Mill: segmented_maxmean
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
	@test gradcheck(x -> sum(PNorm(4)(x, [1:4])), randn(4, 4))
	@test gradcheck(x -> sum(PNorm(4)(x, [1:4, 5:5, 6:8])), randn(4, 8))
	@test gradcheck(x -> sum(segmented_meanmax(x, [1:4])), randn(4, 4))
	@test gradcheck(x -> sum(segmented_meanmax(x, [1:4, 5:5, 6:8])), randn(4, 8))
	@test gradcheck(x -> sum(segmented_maxmean(x, [1:4])), randn(4, 4))
	@test gradcheck(x -> sum(segmented_maxmean(x, [1:4, 5:5, 6:8])), randn(4, 8))

	for g in [	x -> sum(segmented_mean(x, bags1)),
				x -> sum(segmented_max(x, bags1)),
				x -> sum(segmented_meanmax(x, bags1)),
				x -> sum(segmented_maxmean(x, bags1)),
		]
		@test gradcheck(g, x1)
	end
	for g in [  x -> sum(segmented_mean(x, bags2, w)),
				x -> sum(segmented_max(x, bags2, w)),
				x -> sum(segmented_meanmax(x, bags2, w)),
				x -> sum(segmented_maxmean(x, bags2, w))
		]
		@test gradcheck(g, x2)
	end
end

@testset "aggregation functionality" begin
	@test segmented_mean(x2, bags2) ≈ [1.0 4.0 9.0; 2.0 5.0 10.0]
	@test segmented_max(x2, bags2) ≈ [1.0 5.0 11.0; 2.0 6.0 12.0]
	@test segmented_meanmax(x2, bags2) ≈ cat(1, segmented_mean(x2, bags2), segmented_max(x2, bags2))
	@test segmented_maxmean(x2, bags2) ≈ cat(1, segmented_max(x2, bags2), segmented_mean(x2, bags2))
	@test segmented_mean(x2, bags2, w) ≈ [1.0 4.0 236/24; 2.0 5.0 260/24]
	@test segmented_max(x2, bags2, w) ≈ segmented_max(x2, bags2)
	@test segmented_pnorm(x2, bags2, [1, 1], [0, 0]) ≈ segmented_mean(abs.(x2), bags2)
	@test segmented_pnorm(x2, bags2, [2, 2], [0, 0]) ≈ hcat([sqrt.(sum(x2[:, b] .^ 2, 2) ./ length(b)) for b in bags2]...)
	@test segmented_meanmax(x2, bags2, w) ≈ cat(1, segmented_mean(x2, bags2, w), segmented_max(x2, bags2, w))
	@test segmented_maxmean(x2, bags2, w) ≈ cat(1, segmented_max(x2, bags2, w), segmented_mean(x2, bags2, w))
end

# @testset "learning basic pnorm types" begin
# 	model = Mill.AggregationModel(identity, PNorm(1), Flux.Dense(1,2))
# 	loss(x,y) = Flux.logitcrossentropy(model(x).data, Flux.onehotbatch([y], 1:2));
# 	dataset = []
# 	for t in 1:1000
# 		p = 4; c = -2
# 		x = rand(-1.5+c:0.01:1.5+c, 1, 2)
# 		y = sum((abs.(x.-c)) .^ p) .^ (1/p) < 1 ? 1 : 2
# 		push!(dataset, (Mill.BagNode(Mill.ArrayNode(x), [1:2]), y))
# 	end
#
# 	opt = Flux.ADAM(params(model))
# 	Flux.@epochs 30 begin
# 		@show model.a.c
# 		@show p_map(model.a.ρ)
# 		Flux.train!(loss, dataset, opt)
# 		l = 0; r = 0; w = 0
# 		for(X, y) in dataset
# 			pred = findmax(softmax(model(X).data))[2]
# 			if pred == y r += 1 else w += 1 end
# 			l += loss(X, y)
# 		end
# 		@show l
# 		@show r
# 		@show w
# 	end
# end
