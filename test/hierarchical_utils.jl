metadata = fill("metadata", 4)
an1 = ArrayNode(rand(3,4))
b = BagNode(an1, [1:4, 0:-1], metadata)
an2 = ArrayNode(NGramMatrix(["test"], 3, 10, 17))
wb = WeightedBagNode(an2, [1:2,3:4], rand(1:4, 4), metadata)
n1 = TreeNode((b=b,wb=wb))
an3 = ArrayNode(SparseMatrixCSC(rand(10, 2)))
n2 = TreeNode((n1, an3))

n2m = reflectinmodel(n2)
n1m, an3m = n2m.ms
bm, wbm = n1m.ms
an1m = bm.im
an2m = wbm.im

@testset "getindex on strings" begin
    @test n2[""] === n2
    @test n2["E"] === n1
    @test n2["I"] === b
    @test n2["K"] === an1
    @test n2["M"] === wb
    @test n2["O"] === an2
    @test n2["U"] === an3

    @test n2m[""] === n2m
    @test n2m["E"] === n1m
    @test n2m["I"] === bm
    @test n2m["K"] === an1m
    @test n2m["M"] === wbm
    @test n2m["O"] === an2m
    @test n2m["U"] === an3m
end

@testset "children" begin
    @test Set(children(n2)) == Set([n1, an3])
    @test Set(children(n1)) == Set([b, wb])
    @test Set(children(b)) == Set([an1])
    @test Set(children(wb)) == Set([an2])

    @test Set(children(n2m)) == Set([n1m, an3m])
    @test Set(children(n1m)) == Set([bm, wbm])
    @test Set(children(bm)) == Set([an1m])
    @test Set(children(wbm)) == Set([an2m])
end

@testset "nchildren" begin
    @test nchildren(an1) == nchildren(an2) == nchildren(an3) == 0
    @test nchildren(b) == nchildren(wb) == 1
    @test nchildren(n1) == nchildren(n2) == 2

    @test nchildren(an1m) == nchildren(an2m) == nchildren(an3m) == 0
    @test nchildren(bm) == nchildren(wbm) == 1
    @test nchildren(n1m) == nchildren(n2m) == 2
end

@testset "nnodes" begin
    @test nnodes(an1) == nnodes(an2) == nnodes(an3) == 1
    @test nnodes(wb) == nnodes(b) == 2
    @test nnodes(n1) == nnodes(b) + nnodes(wb) + 1
    @test nnodes(n2) == nnodes(n1) + nnodes(an3) + 1

    @test nnodes(an1m) == nnodes(an2m) == nnodes(an3m) == 1
    @test nnodes(wbm) == nnodes(bm) == 2
    @test nnodes(n1m) == nnodes(bm) + nnodes(wbm) + 1
    @test nnodes(n2m) == nnodes(n1m) + nnodes(an3m) + 1
end

@testset "nnodes" begin
    @test nleafs(an1) == nleafs(an2) == nleafs(an3) == 1
    @test nleafs(wb) == nleafs(b) == 1
    @test nleafs(n1) == nleafs(b) + nleafs(wb)
    @test nleafs(n2) == nleafs(n1) + nleafs(an3)

    @test nleafs(an1m) == nleafs(an2m) == nleafs(an3m) == 1
    @test nleafs(wbm) == nleafs(bm) == 1
    @test nleafs(n1m) == nleafs(bm) + nleafs(wbm)
    @test nleafs(n2m) == nleafs(n1m) + nleafs(an3m)
end

@testset "NodeIterator" begin
    @test Set(NodeIterator(n2)) == Set([an1, an2, an3, b, wb, n1, n2])

    @test Set(NodeIterator(n2m)) == Set([an1m, an2m, an3m, bm, wbm, n1m, n2m])
end

@testset "LeafIterator" begin
    @test Set(LeafIterator(n2)) == Set([an1, an2, an3])

    @test Set(LeafIterator(n2m)) == Set([an1m, an2m, an3m])
end

@testset "TypeIterator" begin
    @test Set(TypeIterator{AbstractNode}(n2)) == Set(NodeIterator(n2))
    @test Set(TypeIterator{AbstractBagNode}(n2)) == Set([b, wb])
    @test Set(TypeIterator{AbstractTreeNode}(n2)) == Set([n1, n2])

    @test Set(TypeIterator{MillModel}(n2m)) == Set(NodeIterator(n2m))
    @test Set(TypeIterator{BagModel}(n2m)) == Set([bm, wbm])
    @test Set(TypeIterator{ProductModel}(n2m)) == Set([n1m, n2m])
end

@testset "ZipIterator" begin
    @test ZipIterator(n2, n2m) |> collect == collect(zip(NodeIterator(n2) |> collect, NodeIterator(n2m) |> collect))
end





