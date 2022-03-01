@testset "catobs & getindex operations missing values" begin
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
    Mill.emptyismissing!(true)
    @test  ismissing(x[2].data)
    Mill.emptyismissing!(false)
    @test  nobs(x[2].data) == 0
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

    @test_throws AssertionError BagNode(missing, AlignedBags([1:3]), nothing)
    @test_throws AssertionError BagNode(missing, Mill.ScatteredBags([[1,2,3]]), nothing)
end

@testset "catobs & getindex operations missing values for weighted" begin
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
    Mill.emptyismissing!(true)
    @test  ismissing(x[2].data)
    Mill.emptyismissing!(false)
    @test  nobs(x[2].data) == 0
    @test  x[2].bags.bags == [0:-1]
    @test x[1].weights == [1.0, 0.0, 1.0, 0.5]
    @test x[2].weights == []
    @test  x[1].data.data == a.data.data
    @test  x[1].bags.bags == [1:4]
end

@testset "model operations missing values" begin
    a = BagNode(ArrayNode(rand(3, 4)), [1:4], nothing)
    e = BagNode(missing, AlignedBags([0:-1]), nothing)
    m = BagModel(ArrayModel(Dense(3, 2)), SegmentedMean(2), ArrayModel(Dense(2, 2)))

    @testset "BagNode" begin
        x = reduce(catobs, [a, e])
        @test m(x).data[:, 1] ≈ m(a).data
        @test m(x).data[:, 2] ≈ m(e).data
        @test m(x).data ≈ hcat(m(a).data, m(e).data)
    end

    @testset "ProductNode" begin
        t1 = ProductNode((a, a))
        t2 = ProductNode((a, e))
        t3 = ProductNode((e, a))
        t4 = ProductNode((e, e))
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

@testset "reduction with missing values" begin
    a = ProductNode(x=ArrayNode(rand(3,4)), y=ArrayNode(rand(3,4)))
    b = ProductNode(x=ArrayNode(rand(3,4)), y=ArrayNode(rand(3,4)))

    c = ProductNode(
        x=BagNode(missing, AlignedBags([0:-1]), nothing), y=ArrayNode(rand(3,1)),
    )
    d = ProductNode(
        x=BagNode(a, [1:4], nothing), y=ArrayNode(rand(3,1)),
    )

    ba = BagNode(a, [1:4], nothing)
    bb = BagNode(b, [1:4], nothing)
    bc = BagNode(missing, AlignedBags([0:-1]), nothing)

    bba = BagNode(c, [1:1], nothing)
    bbb = BagNode(d, [1:1], nothing)
    bbc = BagNode(missing, AlignedBags([0:-1]), nothing)

    x = reduce(catobs, [c, d])
    y = reduce(catobs, [ba, bb, bc])
    z = reduce(catobs, [bba, bbb, bbc])

    mx = reflectinmodel(x, d -> Dense(d, 2, relu))
    my = reflectinmodel(y, d -> Dense(d, 2, relu))
    mz = reflectinmodel(z, d -> Dense(d, 2, relu))

    @test mx(x).data[:, 1] ≈ mx(c).data
    @test mx(x).data[:, 2] ≈ mx(d).data

    @test my(y).data[:, 1] ≈ my(ba).data
    @test my(y).data[:, 2] ≈ my(bb).data
    @test my(y).data[:, 3] ≈ my(bc).data

    @test mz(z).data[:, 1] ≈ mz(bba).data
    @test mz(z).data[:, 2] ≈ mz(bbb).data
    @test mz(z).data[:, 3] ≈ mz(bbc).data
end
