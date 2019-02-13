using Mill, Test, Flux
using Mill: ArrayNode, BagNode, TreeNode, catobs

@testset "testing cat & getindex operations missing values" begin
    a = BagNode(ArrayNode(rand(3,4)),[1:4], nothing)
    e = BagNode(nothing, AlignedBags([0:-1]), nothing)

    x = reduce(catobs,[a, e])
    @test x.data.data == a.data.data
    @test x.bags.bags == [1:4, 0:-1]

    x = reduce(catobs,[e, a])
    @test x.data.data == a.data.data
    @test x.bags.bags == [0:-1, 1:4]

    x = reduce(catobs,[e, e])
    @test x.data == nothing
    @test x.bags.bags == [0:-1, 0:-1]

    x = reduce(catobs,[a, e])
    @test  x[2].metadata == nothing
    @test  x[2].data == nothing
    @test  x[2].bags.bags == [0:-1]

    @test  x[1].data.data == a.data.data
    @test  x[1].bags.bags == [1:4]
end

@testset "testing cat & getindex operations missing values" begin
    a = BagNode(ArrayNode(rand(3,4)),[1:4], nothing)
    e = BagNode(nothing, AlignedBags([0:-1]), nothing)

    x = reduce(catobs,[a, e])
    m = BagModel(ArrayModel(Dense(3,2)), SegmentedMean(2), ArrayModel(identity))
    @test m(x).data[:,1] ≈ m(a).data
    @test m(x).data[:,2] ≈ m(e).data
    @test m(x).data ≈ hcat(m(a).data, m(e).data)
end