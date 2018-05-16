using Revise
using Mill
using Base.Test


@testset "length2bags" begin
	@test all(Mill.length2bags([1,3,2]) .== [1:1,2:4,5:6])
	@test all(Mill.length2bags([1,3,0,2]) .== [1:1,2:4,0:-1,5:6])
	@test all(Mill.length2bags([2]) .== [1:2])
	@test all(Mill.length2bags([1]) .== [1:1])
	@test all(Mill.length2bags([0]) .== [0:-1])
end
include("datanode.jl")
include("aggregation.jl")
include("modelnode.jl")