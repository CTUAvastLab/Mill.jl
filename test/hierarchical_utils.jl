using SparseArrays, HierarchicalUtils

metadata = fill("metadata", 4)
an1 = ArrayNode(rand(3,4))
b = BagNode(an1, [1:4, 0:-1], metadata)
an2 = ArrayNode(NGramMatrix(["test", "skunk", "mill", "julia"], 3, 10, 17))
wb = WeightedBagNode(an2, [1:2,3:4], rand(1:4, 4), metadata)
n1 = ProductNode((b=b,wb=wb))
an3 = ArrayNode(SparseMatrixCSC(rand(10, 2)))
n2 = ProductNode((n1, an3))

n2m = reflectinmodel(n2)
n1m, an3m = n2m.ms
bm, wbm = n1m.ms
an1m = bm.im
an2m = wbm.im

struct NumberNode
    n::Int
    chs::Vector{NumberNode}
end
import HierarchicalUtils: NodeType, LeafNode, InnerNode, noderepr, children
NodeType(::Type{NumberNode}) = InnerNode()
noderepr(n::NumberNode) = string(n.n)
children(n::NumberNode) = n.chs

t2 = treemap((n2, n2m)) do (k1, k2), chs
    NumberNode(rand(1:10), collect(chs))
end

@testset "list traversal" begin
    for (n1, n2) in NodeIterator((n2, n2m))
        @test list_traversal(n1) == list_traversal(n2)
    end
end

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
    @test Set(TypeIterator(n2, AbstractNode)) == Set(NodeIterator(n2))
    @test Set(TypeIterator(n2, AbstractBagNode)) == Set([b, wb])
    @test Set(TypeIterator(n2, AbstractProductNode)) == Set([n1, n2])

    @test Set(TypeIterator(n2m, AbstractMillModel)) == Set(NodeIterator(n2m))
    @test Set(TypeIterator(n2m, BagModel)) == Set([bm, wbm])
    @test Set(TypeIterator(n2m, ProductModel)) == Set([n1m, n2m])
end

@testset "Iteration over multiple trees" begin
    @test NodeIterator((n2, n2m)) |> collect == collect(zip(NodeIterator(n2) |> collect, NodeIterator(n2m) |> collect))
end

@testset "printtree" begin
    buf = IOBuffer()
    printtree(buf, n2, trav=true)
    str_repr = String(take!(buf))
    @test str_repr ==
"""
ProductNode [""]
  ├── ProductNode ["E"]
  │     ├─── b: BagNode with 2 bag(s) ["I"]
  │     │         └── ArrayNode(3, 4) ["K"]
  │     └── wb: WeightedNode with 2 bag(s) and weights Σw = 11 ["M"]
  │               └── ArrayNode(17, 4) ["O"]
  └── ArrayNode(10, 2) ["U"]"""

    buf = IOBuffer()
    printtree(buf, n2m, trav=true)
    str_repr = String(take!(buf))
    @test str_repr ==
"""
ProductModel ↦ ArrayModel(Dense(20, 10)) [""]
  ├── ProductModel ↦ ArrayModel(Dense(20, 10)) ["E"]
  │     ├─── b: BagModel ↦ SegmentedMean(10) ↦ ArrayModel(Dense(10, 10)) ["I"]
  │     │         └── ArrayModel(Dense(3, 10)) ["K"]
  │     └── wb: BagModel ↦ SegmentedMean(10) ↦ ArrayModel(Dense(10, 10)) ["M"]
  │               └── ArrayModel(Dense(17, 10)) ["O"]
  └── ArrayModel(Dense(10, 10)) ["U"]"""
end
