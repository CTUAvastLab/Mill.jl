metadata = fill("metadata", 4)
an1 = ArrayNode(rand(Float32, 3, 4))
b = BagNode(an1, [1:4, 0:-1], metadata)
an2 = ArrayNode(NGramMatrix(["test", "skunk", "mill", "julia"], 3, 10, 17))
wb = WeightedBagNode(an2, [1:2,3:4], Float32[1,2,3,4], metadata)
n1 = ProductNode(b=b,wb=wb)
an3 = ArrayNode(SparseMatrixCSC(rand(Float32, 10, 2)))
n2 = ProductNode((n1, an3))

n2m = reflectinmodel(n2)
n1m, an3m = n2m.ms
bm, wbm = n1m.ms
an1m = bm.im
an2m = wbm.im

ss = ["GGGCGGCGA", "CCTCGCGGG", "TTTTCGCTATTTATGAAAATT", "TTCCGGTTTAAGGCGTTTCCG"]
base_arr = collect("ACGT")
possibilities = product(base_arr, base_arr, base_arr) |> collect |> x -> reshape(x, 1, :) .|> (x -> reduce(*, x)) |> x -> x[:]

struct NumberNode
    n::Int
    chs::Vector{NumberNode}
end
import HierarchicalUtils: NodeType, LeafNode, InnerNode, nodeshow, children
NodeType(::Type{NumberNode}) = InnerNode()
nodeshow(io::IO, n::NumberNode) = print(io, n.n)
children(n::NumberNode) = n.chs

function Mill.unpack2mill(ds::LazyNode{:Codons})
    s = ds.data
    ss = map(x -> reduce.(*, partition(x, 3)), s)
    x = reduce(hcat, map(x -> Flux.onehotbatch(x, possibilities), ss))
    BagNode(x, Mill.length2bags(length.(ss)))
end

# specification of printing
NodeType(::Type{<:LazyNode{:Codons}}) = InnerNode()
children(n::LazyNode{:Codons}) = tuple(n.data...)
HierarchicalUtils.@hierarchical_vector
HierarchicalUtils.@primitives

t2 = treemap(n2, n2m) do (k1, k2), chs
    NumberNode(rand(1:10), collect(chs))
end

function buf_printtree(data; kwargs...)
    buf = IOBuffer()
    printtree(buf, data; kwargs...)
    String(take!(buf))
end

@testset "list traversal" begin
    for (n1, n2) in NodeIterator(n2, n2m)
        @test list_traversal(n1) == list_traversal(n2)
    end
end

@testset "getindex on strings" begin
    @test n2[""] ≡ n2
    @test n2["E"] ≡ n1
    @test n2["I"] ≡ b
    @test n2["K"] ≡ an1
    @test n2["M"] ≡ wb
    @test n2["O"] ≡ an2
    @test n2["U"] ≡ an3

    @test n2m[""] ≡ n2m
    @test n2m["E"] ≡ n1m
    @test n2m["I"] ≡ bm
    @test n2m["K"] ≡ an1m
    @test n2m["M"] ≡ wbm
    @test n2m["O"] ≡ an2m
    @test n2m["U"] ≡ an3m
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

@testset "nleafs" begin
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
    @test Set(TypeIterator(AbstractMillNode, n2)) == Set(NodeIterator(n2))
    @test Set(TypeIterator(AbstractBagNode, n2)) == Set([b, wb])
    @test Set(TypeIterator(AbstractProductNode, n2)) == Set([n1, n2])

    @test Set(TypeIterator(AbstractMillModel, n2m)) == Set(NodeIterator(n2m))
    @test Set(TypeIterator(BagModel, n2m)) == Set([bm, wbm])
    @test Set(TypeIterator(ProductModel, n2m)) == Set([n1m, n2m])
end

@testset "Iteration over multiple trees" begin
    @test NodeIterator(n2, n2m) |> collect == collect(zip(NodeIterator(n2) |> collect, NodeIterator(n2m) |> collect))
end

@static if VERSION ≥ v"1.11.0"
    @testset "printtree" begin
        @test buf_printtree(n2, trav=true) ==
            """
            ProductNode [""]  2 obs
              ├── ProductNode ["E"]  2 obs
              │     ├─── b: BagNode ["I"]  2 obs
              │     │         ╰── ArrayNode(3×4 Array with Float32 elements) ["K"]  4 obs
              │     ╰── wb: WeightedBagNode ["M"]  2 obs
              │               ╰── ArrayNode(17×4 NGramMatrix with Int64 elements) ["O"]  4 obs
              ╰── ArrayNode(10×2 SparseMatrixCSC with Float32 elements) ["U"]  2 obs
            """

        @test buf_printtree(n2m, trav=true) ==
            """
            ProductModel ↦ Dense(20 => 10) [""]  2 arrays, 210 params, 928 bytes
              ├── ProductModel ↦ Dense(20 => 10) ["E"]  2 arrays, 210 params, 928 bytes
              │     ├─── b: BagModel ↦ BagCount([SegmentedMean(10); SegmentedMax(10)]) ↦ Dense(21 => 10) ["I"]  4 arrays, 240 params, 1.102 KiB
              │     │         ╰── ArrayModel(Dense(3 => 10)) ["K"]  2 arrays, 40 params, 248 bytes
              │     ╰── wb: BagModel ↦ BagCount([SegmentedMean(10); SegmentedMax(10)]) ↦ Dense(21 => 10) ["M"]  4 arrays, 240 params, 1.102 KiB
              │               ╰── ArrayModel(Dense(17 => 10)) ["O"]  2 arrays, 180 params, 808 bytes
              ╰── ArrayModel(Dense(10 => 10)) ["U"]  2 arrays, 110 params, 528 bytes
            """
    end
end

@static if VERSION ≥ v"1.11.0"
    @testset "LazyNode" begin
        ds = LazyNode{:Codons}(ss)
        m = Mill.reflectinmodel(ds)
        @test nchildren(ds) == 4
        @test nleafs(ds) == 4
        @test nnodes(ds) == 5

        @test buf_printtree(ds, trav=true) ==
            """
            LazyNode{Codons}(String) [""]  4 obs
              ├── "GGGCGGCGA" ["6"]
              ├── "CCTCGCGGG" ["E"]
              ├── "TTTTCGCTATTTATGAAAATT" ["M"]
              ╰── "TTCCGGTTTAAGGCGTTTCCG" ["U"]
            """

        @test buf_printtree(m, trav=true) ==
            """
            LazyModel{Codons} [""]
              ╰── BagModel ↦ BagCount([SegmentedMean(10); SegmentedMax(10)]) ↦ Dense(21 => 10) ["U"]  4 arrays, 240 params, 1.102 KiB
                    ╰── ArrayModel(Dense(64 => 10)) ["k"]  2 arrays, 650 params, 2.625 KiB
            """
    end
end
