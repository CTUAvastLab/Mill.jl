using Mill, Test
using Mill: MissingNode, ArrayNode

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