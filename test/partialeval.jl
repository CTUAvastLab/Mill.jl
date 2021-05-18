@testset "partialeval" begin
    metadata = fill("metadata", 4)
    an1 = ArrayNode(rand(3,4))
    b = BagNode(an1, [1:4, 0:-1], metadata)
    an2 = ArrayNode(randn(5, 4))
    wb = WeightedBagNode(an2, [1:2,3:4], rand(4), metadata)
    pn = ProductNode((b=b,wb=wb))
    an3 = ArrayNode(rand(10, 2))
    ds = ProductNode((pn, an3))
    # printtree(ds)
    m = reflectinmodel(ds, d -> Chain(Dense(d, 4, relu), Dense(4,3)), d -> SegmentedMeanMax(d))


    @test partialeval(m.ms[2], ds.data[2], an3)[1] === m.ms[2]
    @test partialeval(m.ms[2], ds.data[2], an3)[2] === ds.data[2]
    @test partialeval(m.ms[2], ds.data[2], an1)[1] === identity_model()
    @test partialeval(m.ms[2], ds.data[2], an1)[2].data ≈ m.ms[2](ds.data[2]).data
    tm, td = partialeval(m.ms[2], ds.data[2], an3)
    @test tm(td).data ≈ m.ms[2](ds.data[2]).data
    @test partialeval(m.ms[2], ds.data[2], an1)[2].data ≈ m.ms[2](ds.data[2]).data

    @test partialeval(m.ms[1][:b], ds.data[1][:b], an1)[1] === m.ms[1][:b]
    @test partialeval(m.ms[1][:b], ds.data[1][:b], an1)[2] === ds.data[1][:b]
    @test partialeval(m.ms[1][:b], ds.data[1][:b], b)[1] === m.ms[1][:b]
    @test partialeval(m.ms[1][:b], ds.data[1][:b], b)[2] === ds.data[1][:b]
    tm, td = partialeval(m.ms[1][:b], ds.data[1][:b], an3)
    @test tm(td).data ≈ m.ms[1][:b](ds.data[1][:b]).data
    @test partialeval(m.ms[1][:b], ds.data[1][:b], an3)[1] === identity_model()
    @test partialeval(m.ms[1][:b], ds.data[1][:b], an3)[2].data ≈ m.ms[1][:b](ds.data[1][:b]).data

    @test partialeval(m.ms[1][:wb], ds.data[1][:wb], an2)[1] === m.ms[1][:wb]
    @test partialeval(m.ms[1][:wb], ds.data[1][:wb], an2)[2] === ds.data[1][:wb]
    @test partialeval(m.ms[1][:wb], ds.data[1][:wb], wb)[1] === m.ms[1][:wb]
    @test partialeval(m.ms[1][:wb], ds.data[1][:wb], wb)[2] === ds.data[1][:wb]
    tm, td = partialeval(m.ms[1][:wb], ds.data[1][:wb], an3)
    @test tm(td).data ≈ m.ms[1][:wb](ds.data[1][:wb]).data
    @test partialeval(m.ms[1][:wb], ds.data[1][:wb], an3)[1] === identity_model()
    @test partialeval(m.ms[1][:wb], ds.data[1][:wb], an3)[2].data ≈ m.ms[1][:wb](ds.data[1][:wb]).data


    @test partialeval(m.ms[1], ds.data[1], an1)[1].ms[1] === m.ms[1].ms[1]
    @test partialeval(m.ms[1], ds.data[1], an1)[2].data[1] === ds.data[1].data[1]
    @test partialeval(m.ms[1], ds.data[1], an1)[1].ms[2] === identity_model()
    @test partialeval(m.ms[1], ds.data[1], an1)[2].data[2].data ≈ m.ms[1].ms[2](ds.data[1].data[2]).data
    @test partialeval(m.ms[1], ds.data[1], an3)[1] === identity_model()
    @test partialeval(m.ms[1], ds.data[1], an3)[2].data ≈ m.ms[1](ds.data[1]).data
    tm, td = partialeval(m.ms[1], ds.data[1], an1)
    @test tm(td).data ≈ m.ms[1](ds.data[1]).data
    @test partialeval(m.ms[2], ds.data[2], an1)[2].data ≈ m.ms[2](ds.data[2]).data

    @test partialeval(m, ds, 1)[1] === identity_model()
    @test partialeval(m, ds, 1)[2].data ≈ m(ds).data
    tm, td = partialeval(m, ds, an1)
    @test tm(td).data ≈ m(ds).data
end
    
