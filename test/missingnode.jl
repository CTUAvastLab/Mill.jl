using Mill, Test, Flux, FiniteDifferences
using Mill: MissingNode, ArrayNode, MissingModel
using Finite

@testset "missingnode" begin 
	d = ArrayNode([1 2 3; 4 5 6])
	x = MissingNode(d)
	@test x[[1,3]].data.data ≈ d[[1,3]].data
	@test x[[2]].data.data ≈ d[[2]].data

	d = ArrayNode([1 3; 4 6])
	x = MissingNode(d,[true,false,true])
	@test x[[1,3]].data.data ≈ d[[1,2]].data
	@test x[[1,2]].data.data ≈ d[[1]].data
	@test x[[2,3]].data.data ≈ d[[2]].data
	@test isempty(x[[2]].data.data)

	a, b, c = x[[1]],x[[2]],x[[3]]
	@test catobs(a,b).data.data ≈ a.data.data
	@test catobs(a,b).present ≈ [true,false]
	@test catobs(a,c).data.data ≈ d.data
	@test catobs(a,c).present ≈ [true,true]
	@test catobs(b,c).data.data ≈ c.data.data
	@test catobs(b,c).present ≈ [false,true]
	@test isempty(catobs(b,b).data.data)
	@test catobs(b,b).present ≈ [false,false]
	@test catobs(b,c,b).data.data ≈ c.data.data
	@test catobs(b,c,b).present ≈ [false,true,false]
end

@testset "missingmodel" begin 
	xx = randn(2,3)
	d = ArrayNode(xx)
	x = f64(MissingNode(d));
	m = f64(MissingModel(ArrayModel(Dense(2,3)), [1, -1, 0]));

	for present_mask in  [[true,true,true], [true,false,true], [false,true,false], [true,true,false], [false,true,true]]
		present_idxs = setdiff(1:3, findall(present_mask))
		x = MissingNode(d[present_mask], present_mask);
		@test m(x)[present_mask].data ≈ m.m(x.data).data
		@test gradient(xx -> sum(m(MissingNode(ArrayNode(xx))).data), xx)[1] ≈
		grad(central_fdm(5, 1), xx -> sum(m(MissingNode(ArrayNode(xx))).data), xx)[1]

		if !all(present_mask) 
			ps = Flux.params(m)
			@test all(gradient(() -> sum(m(x).data), ps)[m.θ] .== 3 - sum(present_mask))
			@test m(x)[.!present_mask].data ≈ repeat(m.θ, 1, 3 - sum(present_mask))
		end
	end

	d = ArrayNode([1 2 3; 4 5 6])
	x = MissingNode(d)
	_reflectinmodel(x, d -> Dense(d, 3), d -> SegmentedMean)
end

