using Base.Test
import NestedMill: ExtractScalar, ExtractCategorical, ExtractArray, ExtractBranch


@testset "Testing scalar conversion" begin
	sc = ExtractScalar(Float64,2,3)
	@test sc("5") == 9
	@test sc(5) == 9
end


@testset "Testing categorical conversion to one hot" begin
	sc = ExtractCategorical(Float64,2:4)
	@test all(sc(2) .== [1,0,0])
	@test all(sc(3) .== [0,1,0])
	@test all(sc(4) .== [0,0,1])
	@test all(sc(5) .== [0,0,0])
end


@testset "Testing array conversion" begin
	sc = ExtractArray(ExtractCategorical(Float64,2:4))
	@test all(sc([2,3,4]).data .== eye(3))
	sc = ExtractArray(ExtractScalar(Float64))
	@test all(sc([2,3,4]).data .== [2 3 4])
end


@testset "Testing ExtractBranch" begin
	vec = Dict("a" => ExtractScalar(Float64,2,3),"b" => ExtractScalar(Float64));
	other = Dict("c" => ExtractArray(ExtractScalar(Float64,2,3)));
	br = ExtractBranch(Float64,vec,other)
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


	br = ExtractBranch(Float64,vec,nothing)
	a1 = br(Dict("a" => 5, "b" => 7, "c" => [1,2,3,4]))
	a2 = br(Dict("a" => 5, "b" => 7))
	a3 = br(Dict("a" => 5, "c" => [1,2,3,4]))
	@test all(a1.data .==[7; 9])
	@test all(a2.data .==[7; 9])
	@test all(a3.data .==[0; 9])
	
	
	br = ExtractBranch(Float64,nothing,other)
	a1 = br(Dict("a" => 5, "b" => 7, "c" => [1,2,3,4]))
	a2 = br(Dict("a" => 5, "b" => 7))
	a3 = br(Dict("a" => 5, "c" => [1,2,3,4]))

	@test all(a1.data.data .== [-3 0 3 6])
	@test all(a1.data.bags .== [1:4])
	@test all(cat(a1,a1).data.data .== [-3 0 3 6 -3 0 3 6])
	@test all(cat(a1,a1).data.bags .== [1:4,5:8])

	@test all(cat(a1,a2).data.data .== [-3 0 3 6])
	@test all(cat(a1,a2).data.bags .== [1:4,0:-1])
	

	@test all(a3.data.data .== [-3 0 3 6])
	@test all(a3.data.bags .== [1:4])
	@test all(cat(a3,a3).data.data .== [-3 0 3 6 -3 0 3 6])
	@test all(cat(a3,a3).data.bags .== [1:4,5:8])

end


@testset "Testing Nested Missing Arrays" begin
	other = Dict("a" => ExtractArray(ExtractScalar(Float64,2,3)),"b" => ExtractArray(ExtractScalar(Float64,2,3)));
	br = ExtractBranch(Float64,vec,other)
	a1 = br(Dict("a" => [1,2,3], "b" => [1,2,3,4]))
	a2 = br(Dict("b" => [2,3,4]))
	a3 = br(Dict("a" => [2,3,4]))
	a4 = br(Dict{String,Any})
end


brtext = """
{	"type": "ExtractBranch",
  "vec": {
    "a": {"type": "ExtractScalar", "center": 2, "scale": 3},
    "b": {"type": "ExtractScalar", "center": 0.0, "scale": 1.0}
  },
  "other": {
    "c": { "type": "ExtractArray",
      "item": {
        "type": "ExtractScalar",
        "center": 2,
        "scale": 3
      }
    }
  }
}
"""

import NestedMill: tojson, interpret
@testset "testing interpreting and exporting to JSON" begin
	br = interpret(brtext)
	@test tojson(interpret(tojson(br))) == tojson(br)
end