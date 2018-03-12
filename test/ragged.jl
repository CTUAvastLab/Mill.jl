@testset "creating bags" begin
	k = [2, 2, 2, 1, 1, 3]
	@test all(NestedMill.bag(k) .== [1:3,4:5,6:6])
end


@testset "testing remapping for subsets" begin
  @test all(NestedMill.remapbag([1:1,2:3,4:5],[2,3])[1] .== [1:2,3:4])
  @test all(NestedMill.remapbag([1:1,2:3,4:5],[2,3])[2] .== [2,3,4,5])
  @test all(NestedMill.remapbag([1:2,3:3,4:5],[1,3])[1] .== [1:2,3:4])
  @test all(NestedMill.remapbag([1:2,3:3,4:5],[1,3])[2] .== [1,2,4,5])
  @test all(NestedMill.remapbag([1:2,3:3,4:5],[2,3])[1] .== [1:1,2:3])
  @test all(NestedMill.remapbag([1:2,3:3,4:5],[2,3])[2] .== [3,4,5])

  @test all(NestedMill.remapbag([1:2,0:-1,3:4],[2,3])[1] .== [0:-1,1:2])
  @test all(NestedMill.remapbag([1:2,0:-1,3:4],[2,3])[2] .== [3,4])
end


