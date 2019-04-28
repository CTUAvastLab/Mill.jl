using Mill, Flux
using Mill: ArrayNode, BagNode, TreeNode, catobs

@testset "testing catobs & getindex operations missing values" begin
    a = BagNode(ArrayNode(rand(3,4)), [1:4], nothing)
    e = BagNode(missing, AlignedBags([0:-1]), nothing)

    x = reduce(catobs, [a, e])
    @test x.data.data == a.data.data
    @test x.bags.bags == [1:4, 0:-1]

    x = reduce(catobs, [e, a])
    @test x.data.data == a.data.data
    @test x.bags.bags == [0:-1, 1:4]

    x = reduce(catobs, [e, e])
    @test ismissing(x.data)
    @test x.bags.bags == [0:-1, 0:-1]

    x = reduce(catobs, [a, e])
    @test  isnothing(x[2].metadata)
    @test  ismissing(x[2].data)
    @test  x[2].bags.bags == [0:-1]

    @test  x[1].data.data == a.data.data
    @test  x[1].bags.bags == [1:4]

    @test catobs(a, e).data.data == a.data.data
    @test catobs(a, e).bags.bags == [1:4, 0:-1]

    a = BagNode(a, [1:1], missing)
    @test catobs(a, e).bags.bags == [1:1, 0:-1]
    @test catobs(e, a).bags.bags == [0:-1, 1:1]
    @test catobs(e, a).data.data.data == a.data.data.data

    @test catobs(a,e,a,e).data.data.data == catobs(a,a).data.data.data
    @test catobs(a,e,a,e).bags.bags == [1:1, 0:-1, 2:2, 0:-1]

    @test_throws ErrorException BagNode(missing, AlignedBags([1:3]), nothing)
    @test_throws ErrorException BagNode(missing, Mill.ScatteredBags([[1,2,3]]), nothing)
end

@testset "testing catobs & getindex operations missing values for weighted" begin
    a = WeightedBagNode(ArrayNode(rand(3,4)), [1:4], [1.0, 0.0, 1.0, 0.5], nothing)
    e = WeightedBagNode(missing, AlignedBags([0:-1]), [], nothing)

    x = reduce(catobs, [a, e])
    @test x.data.data == a.data.data
    @test x.bags.bags == [1:4, 0:-1]
    @test x.weights == [1.0, 0.0, 1.0, 0.5]

    x = reduce(catobs, [e, a])
    @test x.data.data == a.data.data
    @test x.bags.bags == [0:-1, 1:4]
    @test x.weights == [1.0, 0.0, 1.0, 0.5]

    x = reduce(catobs, [e, e])
    @test ismissing(x.data)
    @test x.bags.bags == [0:-1, 0:-1]
    @test x.weights == []

    x = reduce(catobs, [a, e])
    @test  isnothing(x[2].metadata)
    @test  ismissing(x[2].data)
    @test  x[2].bags.bags == [0:-1]
    @test x[1].weights == [1.0, 0.0, 1.0, 0.5]
    @test x[2].weights == []
    @test  x[1].data.data == a.data.data
    @test  x[1].bags.bags == [1:4]
end

@testset "testing model operations missing values" begin
    a = BagNode(ArrayNode(rand(3,4)), [1:4], nothing)
    e = BagNode(missing, AlignedBags([0:-1]), nothing)
    m = BagModel(ArrayModel(Dense(3, 2)), SegmentedMean(2), ArrayModel(Dense(2, 2)))

    @testset "BagNode" begin
        x = reduce(catobs, [a, e])
        @test m(x).data[:, 1] ≈ m(a).data
        @test m(x).data[:, 2] ≈ m(e).data
        @test m(x).data ≈ hcat(m(a).data, m(e).data)
    end

    @testset "TreeNode" begin
        t1 = TreeNode((a, a))
        t2 = TreeNode((a, e))
        t3 = TreeNode((e, a))
        t4 = TreeNode((e, e))
        tt = [t1, t2, t3, t4]
        x  = reduce(catobs, tt)
        tm = ProductModel((m, m))
        o = tm(x).data
        for i in 1:length(tt)
            @test o[:,i] ≈ tm(tt[i]).data
        end
        for i in 1:length(tt)
            @test o[:,i] ≈ tm(x[i]).data
        end
    end
end
