@testset "findnonempty" begin
end

@testset "ModelLens" begin
end

@testset "replacein" begin
    metadata = fill("metadata", 4)
    an1 = ArrayNode(rand(3,4))
    b = BagNode(an1, [1:4, 0:-1], metadata)
    an2 = ArrayNode(randn(5, 4))
    wb = WeightedBagNode(an2, [1:2,3:4], Vector{Float32}(rand(1:4, 4)), metadata)
    pn = ProductNode((b=b,wb=wb))
    an3 = ArrayNode(rand(10, 2))
    x1 = ProductNode((pn, an3))

    t = list_traversal(x1)
    for t in list_traversal(x1)
        x2 = deepcopy(x1)
        x3 = replacein(x2, x2[t], x1[t])
        @test x3[t] === x1[t]
    end

    m1 = reflectinmodel(x1)
    for t in list_traversal(m1)
        m2 = deepcopy(m1)
        m3 = replacein(m2, m2[t], m1[t])
        @test m3[t] === m1[t]
    end

end

@testset "findin" begin
    metadata = fill("metadata", 4)
    an1 = ArrayNode(rand(3,4))
    b = BagNode(an1, [1:4, 0:-1], metadata)
    an2 = ArrayNode(randn(5, 4))
    wb = WeightedBagNode(an2, [1:2,3:4], Vector{Float32}(rand(1:4, 4)), metadata)
    pn = ProductNode((b=b,wb=wb))
    an3 = ArrayNode(rand(10, 2))
    x1 = ProductNode((a = pn, b = an3))

    t = list_traversal(x1)
    for t in list_traversal(x1)
        l = findin(x1, x1[t])
        @test get(x1,l) === x1[t]
    end

    m1 = reflectinmodel(x1)
    for t in list_traversal(m1)
        l = findin(m1,  m1[t])
        @test get(m1, l) === m1[t]
    end
end
