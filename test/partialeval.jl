@testset "partialeval" begin
    metadata = fill("metadata", 4)
    an1 = ArrayNode(rand(3,4))
    b = BagNode(an1, [1:4, 0:-1], metadata)
    an2 = ArrayNode(randn(5, 4))
    wb = WeightedBagNode(an2, [1:2,3:4], rand(4), metadata)
    pn = ProductNode(; b, wb)
    an3 = ArrayNode(rand(10, 2))
    ds = ProductNode((pn, an3))
    m = reflectinmodel(ds, d -> Chain(Dense(d, 4, relu), Dense(4,3)), SegmentedMeanMax)

    mm = m.ms[2]
    dd = ds.data[2]
    @test partialeval(mm, dd, an3)[1] ≡ mm
    @test partialeval(mm, dd, an3)[2] ≡ dd
    @test partialeval(mm, dd, an1)[1] ≡ identity
    @test partialeval(mm, dd, an1)[2] ≈ mm(dd)
    tm, td = partialeval(mm, dd, an3)
    @test tm(td) ≈ mm(dd)
    @test partialeval(mm, dd, an1)[2] ≈ mm(dd)

    mm = m.ms[1][:b]
    dd = ds.data[1][:b]
    @test partialeval(mm, dd, an1)[1] ≡ mm
    @test partialeval(mm, dd, an1)[2] ≡ dd
    @test partialeval(mm, dd, b)[1] ≡ mm
    @test partialeval(mm, dd, b)[2] ≡ dd
    tm, td = partialeval(mm, dd, an3)
    @test tm(td) ≈ mm(dd)
    @test partialeval(mm, dd, an3)[1] ≡ identity
    @test partialeval(mm, dd, an3)[2] ≈ mm(dd)

    mm = m.ms[1][:wb]
    dd = ds.data[1][:wb]
    @test partialeval(mm, dd, an2)[1] ≡ mm
    @test partialeval(mm, dd, an2)[2] ≡ dd
    @test partialeval(mm, dd, wb)[1] ≡ mm
    @test partialeval(mm, dd, wb)[2] ≡ dd
    tm, td = partialeval(mm, dd, an3)
    @test tm(td) ≈ mm(dd)
    @test partialeval(mm, dd, an3)[1] ≡ identity
    @test partialeval(mm, dd, an3)[2] ≈ mm(dd)

    mm = m.ms[1]
    dd = ds.data[1]
    @test partialeval(mm, dd, an1)[1].ms[1] ≡ mm.ms[1]
    @test partialeval(mm, dd, an1)[2].data[1] ≡ dd.data[1]
    @test partialeval(mm, dd, an1)[1].ms[2] ≡ ArrayModel(identity)
    @test partialeval(mm, dd, an1)[2].data[2].data ≈ mm.ms[2](dd.data[2])
    @test partialeval(mm, dd, an3)[1] ≡ identity
    @test partialeval(mm, dd, an3)[2] ≈ mm(dd)
    tm, td = partialeval(mm, dd, an1)
    @test tm(td) ≈ mm(dd)
    @test partialeval(m.ms[2], ds.data[2], an1)[2] ≈ m.ms[2](ds.data[2])

    @test partialeval(m, ds, 1)[1] ≡ identity
    @test partialeval(m, ds, 1)[2] ≈ m(ds)
    tm, td = partialeval(m, ds, an1)
    @test tm(td) ≈ m(ds)
end
    
