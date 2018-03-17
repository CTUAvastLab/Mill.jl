using Base.Test
import NestedMill: Scalar, Categorical, ArrayOf, Branch


@testset "Testing scalar conversion" begin
	sc = Scalar(Float64,2,3)
	@test sc("5") == 9
	@test sc(5) == 9
end


@testset "Testing categorical conversion to one hot" begin
	sc = Categorical(Float64,2:4)
	@test all(sc(2) .== [1,0,0])
	@test all(sc(3) .== [0,1,0])
	@test all(sc(4) .== [0,0,1])
	@test all(sc(5) .== [0,0,0])
end


@testset "Testing array conversion" begin
	sc = ArrayOf(Categorical(Float64,2:4))
	@test all(sc([2,3,4]).data .== eye(3))
	sc = ArrayOf(Scalar())
	@test all(sc([2,3,4]).data .== [2 3 4])
end


@testset "Testing Branch" begin
	vec = Dict("a" => Scalar(Float64,2,3),"b" => Scalar());
	other = Dict("c" => ArrayOf(Scalar(Float64,2,3)));
	br = Branch(Float64,vec,other)
	a1 = br(Dict("a" => 5, "b" => 7, "c" => [1,2,3,4]))
	a2 = br(Dict("a" => 5, "b" => 7))
	a3 = br(Dict("a" => 5, "c" => [1,2,3,4]))
	@test all(cat(a1,a1).data[1] .==[7 7; 9 9])
	@test all(cat(a1,a1).data[2].data .== [-3 0 3 6 -3 0 3 6])
	@test all(cat(a1,a1).data[2].bags .== [1:4,5:8])
	
	@test all(cat(a1,a2).data[1] .==[7 7; 9 9])
	@test all(cat(a1,a2).data[2].data .== [-3 0 3 6])
	@test all(cat(a1,a2).data[2].bags .== [1:4,0:-1])

	@test all(cat(a2,a3).data[1] .==[7 0; 9 9])
	@test all(cat(a2,a3).data[2].data .== [-3 0 3 6])
	@test all(cat(a2,a3).data[2].bags .== [0:-1,1:4])

	@test all(cat(a1,a3).data[1] .==[7 0; 9 9])
	@test all(cat(a1,a3).data[2].data .== [-3 0 3 6 -3 0 3 6])
	@test all(cat(a1,a3).data[2].bags .== [1:4,5:8])
end