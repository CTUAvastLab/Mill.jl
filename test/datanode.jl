@testset "creating bags" begin
    k = [2, 2, 2, 1, 1, 3]
    @test Mill.bags(k).bags == [1:3,4:5,6:6]
end

metadata = fill("metadata", 4)
a = BagNode(ArrayNode(rand(3, 4)),[1:4], metadata)
b = BagNode(ArrayNode(rand(3, 4)),[1:2, 3:4], metadata)
c = BagNode(ArrayNode(rand(3, 4)),[1:1, 2:2, 3:4], metadata)
d = BagNode(ArrayNode(rand(3, 4)),[1:4, 0:-1], metadata)
wa = WeightedBagNode(ArrayNode(rand(3, 4)),[1:4], rand(1:4, 4), metadata)
wb = WeightedBagNode(ArrayNode(rand(3, 4)),[1:2, 3:4], rand(1:4, 4), metadata)
wc = WeightedBagNode(ArrayNode(rand(3, 4)),[1:1, 2:2, 3:4], rand(1:4, 4), metadata)
wd = WeightedBagNode(ArrayNode(rand(3, 4)),[1:4, 0:-1], rand(1:4, 4), metadata)
e = ArrayNode(rand(2, 2))

f = ProductNode((wb,b))
g = ProductNode([c, wc])
h = ProductNode((wc,c))
i = ProductNode((
              b,
              ProductNode((
                        b,
                        BagNode(
                                BagNode(
                                        ArrayNode(rand(2, 4)),
                                        [1:1, 2:2, 3:3, 4:4]
                                       ),
                                [1:3, 4:4]
                               )
                       ))
             ))
k = ProductNode((a = wb, b = b))
l = ProductNode((a = wc, b = c))
m = ProductNode((a = wc, c = c))

@testset "testing nobs" begin
    @test nobs(a) == nobs(wa) == 1
    @test nobs(b) == nobs(wb) == 2
    @test nobs(c) == nobs(wc) == 3
    @test nobs(d) == nobs(wd) == 2
    @test nobs(e) == 2
    @test nobs(f) == nobs(wb) == nobs(b) == nobs(k)
    @test nobs(g) == nobs(c) == nobs(wc) == nobs(l)
    @test nobs(h) == nobs(wc) == nobs(c)
    @test nobs(i) == nobs(b)
end

@testset "testing ArrayNode hcat and vcat" begin
    @test catobs(e, e).data == hcat(e.data, e.data) == reduce(catobs, [e,e]).data
    @test hcat(e, e).data == hcat(e.data, e.data)
    @test vcat(e, e).data == vcat(e.data, e.data)
    x = ArrayNode(randn(2,3), rand(2, 3))
    @test catobs(x, x[0:-1]) isa ArrayNode{Array{Float64,2},Array{Float64,2}}
    @inferred catobs(x, x[0:-1])
    @test reduce(catobs, [x, x[0:-1]]) isa ArrayNode{Array{Float64,2},Array{Float64,2}}
    @inferred reduce(catobs, [x, x[0:-1]])
    @test cat(e, e, dims = ndims(e)).data == hcat(e.data, e.data)
end

@testset "testing BagNode catobs" begin
    @test catobs(a, b, c).data.data == hcat(a.data.data, b.data.data, c.data.data)
    @test reduce(catobs, [a, b, c]).data.data == hcat(a.data.data, b.data.data, c.data.data)
    @test catobs(a, b, c).bags.bags == vcat(a.bags, b.bags, c.bags).bags
    @test reduce(catobs, [a, b, c]).bags.bags == vcat(a.bags, b.bags, c.bags).bags
    @test catobs(c, a).data.data == hcat(c.data.data, a.data.data)
    @test reduce(catobs, [c, a]).data.data == hcat(c.data.data, a.data.data)
    @test catobs(c, a).bags.bags == vcat(c.bags, a.bags).bags
    @test reduce(catobs, [c, a]).bags.bags == vcat(c.bags, a.bags).bags
    @test catobs(a, c).data.data == hcat(a.data.data, c.data.data)
    @test reduce(catobs, [a, c]).data.data == hcat(a.data.data, c.data.data)
    @test catobs(a, c).bags.bags == vcat(a.bags, c.bags).bags
    @test reduce(catobs, [a, c]).bags.bags == vcat(a.bags, c.bags).bags
    @test catobs(a, d).data.data == hcat(a.data.data, d.data.data)
    @test reduce(catobs, [a, d]).data.data == hcat(a.data.data, d.data.data)
    @test catobs(a, d).bags.bags == vcat(a.bags, d.bags).bags
    @test reduce(catobs, [a, d]).bags.bags == vcat(a.bags, d.bags).bags
    @test catobs(d, a).data.data == hcat(d.data.data, a.data.data)
    @test reduce(catobs, [d, a]).data.data == hcat(d.data.data, a.data.data)
    @test catobs(d, a).bags.bags == vcat(d.bags, a.bags).bags
    @test reduce(catobs, [d, a]).bags.bags == vcat(d.bags, a.bags).bags
    @test catobs(d).data.data == hcat(d.data.data)
    @test reduce(catobs, [d]).data.data == hcat(d.data.data)
    @test catobs(d).bags.bags == vcat(d.bags).bags
    @test reduce(catobs, [d]).bags.bags == vcat(d.bags).bags
    @test cat(a, b, dims = ndims(a)).data.data == hcat(a.data.data, b.data.data)
end

@testset "testing WeightedBagNode catobs" begin
    @test catobs(wa, wb, wc).data.data == hcat(wa.data.data, wb.data.data, wc.data.data)
    @test reduce(catobs, [wa, wb, wc]).data.data == hcat(wa.data.data, wb.data.data, wc.data.data)
    @test catobs(wa, wb, wc).bags.bags == vcat(wa.bags, wb.bags, wc.bags).bags
    @test reduce(catobs, [wa, wb, wc]).bags.bags == vcat(wa.bags, wb.bags, wc.bags).bags
    @test catobs(wc, wa).data.data == hcat(wc.data.data, wa.data.data)
    @test reduce(catobs, [wc, wa]).data.data == hcat(wc.data.data, wa.data.data)
    @test catobs(wc, wa).bags.bags == vcat(wc.bags, wa.bags).bags
    @test reduce(catobs, [wc, wa]).bags.bags == vcat(wc.bags, wa.bags).bags
    @test catobs(wa, wc).data.data == hcat(wa.data.data, wc.data.data)
    @test reduce(catobs, [wa, wc]).data.data == hcat(wa.data.data, wc.data.data)
    @test catobs(wa, wc).bags.bags == vcat(wa.bags, wc.bags).bags
    @test reduce(catobs, [wa, wc]).bags.bags == vcat(wa.bags, wc.bags).bags
    @test catobs(wa, wd).data.data == hcat(wa.data.data, wd.data.data)
    @test reduce(catobs, [wa, wd]).data.data == hcat(wa.data.data, wd.data.data)
    @test catobs(wa, wd).bags.bags == vcat(wa.bags, wd.bags).bags
    @test reduce(catobs, [wa, wd]).bags.bags == vcat(wa.bags, wd.bags).bags
    @test catobs(wd, wa).data.data == hcat(wd.data.data, wa.data.data)
    @test reduce(catobs, [wd, wa]).data.data == hcat(wd.data.data, wa.data.data)
    @test catobs(wd, wa).bags.bags == vcat(wd.bags, wa.bags).bags
    @test reduce(catobs, [wd, wa]).bags.bags == vcat(wd.bags, wa.bags).bags
    @test catobs(wd).data.data == hcat(wd.data.data)
    @test reduce(catobs, [wd]).data.data == hcat(wd.data.data)
    @test catobs(wd).bags.bags == vcat(wd.bags).bags
    @test reduce(catobs, [wd]).bags.bags == vcat(wd.bags).bags
end

@testset "testing catobs with missing values" begin
    @test catobs(a, b, c).data.data == catobs(a, b, missing, c).data.data
    @test catobs(a, b, c).bags.bags == catobs(a, b, missing, c).bags.bags
    @test catobs(wa, wb, wc).data.data == catobs(wa, wb, missing, wc).data.data
    @test catobs(wa, wb, wc).bags.bags == catobs(wa, wb, missing, wc).bags.bags
    @test catobs(wa, wb, wc).weights == catobs(wa, wb, missing, wc).weights
end

@testset "testing catobs stability" begin
    for n in [a, b, c, d, e, f, h, k, l, wa, wb, wc, wd]
        @inferred catobs(n, n)
        @inferred reduce(catobs, [n, n])
    end
end

@testset "testing hierarchical catobs on product nodes" begin
    @test catobs(f, h).data[1].data.data == hcat(wb.data.data, wc.data.data)
    @test reduce(catobs, [f, h]).data[1].data.data == hcat(wb.data.data, wc.data.data)
    @test catobs(f, h).data[2].data.data == hcat(b.data.data, c.data.data)
    @test reduce(catobs, [f, h]).data[2].data.data == hcat(b.data.data, c.data.data)
    @test catobs(f, h, f).data[1].data.data == hcat(wb.data.data, wc.data.data, wb.data.data)
    @test reduce(catobs, [f, h, f]).data[1].data.data == hcat(wb.data.data, wc.data.data, wb.data.data)

    @test catobs(k, l).data[1].data.data == hcat(wb.data.data, wc.data.data)
    @test catobs(k, l).data[2].data.data == hcat(b.data.data, c.data.data)
    @test nobs(catobs(k,l)) == nobs(k) + nobs(l)

    # different tuple length or keys or content
    @test_throws ArgumentError catobs(g, i)
    @test_throws ArgumentError reduce(catobs, [g, i])
    @test_throws ArgumentError catobs(k, m)
    @test_throws ArgumentError reduce(catobs, [k, m])
    @test_throws ArgumentError catobs(l, m)
    @test_throws ArgumentError reduce(catobs, [l, m])
    @test_throws ArgumentError catobs(f, g)
    @test_throws ArgumentError reduce(catobs, [f, g])
    @test_throws MethodError catobs(f, i)
    @test_throws MethodError reduce(catobs, [f, i])
end

@testset "testing BagNode indexing" begin
    @test a[1].data.data == a.data.data
    @test a[1].bags.bags == [1:4]
    @test b[1:2].data.data == b.data.data
    @test b[1:2].bags.bags == [1:2,3:4]
    @test b[2].data.data == b.data.data[:,3:4]
    @test b[2].bags.bags == [1:2]
    @test b[1].data.data == b.data.data[:,1:2]
    @test b[1].bags.bags == [1:2]
    @test c[1].data.data == c.data.data[:,1:1]
    @test c[1].bags.bags == [1:1]
    @test c[[1,2]].data.data == c.data.data[:,1:2]
    @test c[[1,2]].bags.bags == [1:1,2:2]
    @test c[[2,1]].data.data == c.data.data[:,[2,1]]
    @test c[[2,1]].bags.bags == [1:1,2:2]
    @test d[[2,1]].data.data == d.data.data
    @test d[[2,1]].bags.bags == [0:-1,1:4]
    @test d[1:2].data.data == d.data.data
    @test d[1:2].bags.bags == [1:4,0:-1]
    @test nobs(d[2].data) == 0
    @test d[2].bags.bags == [0:-1]
    @test isempty(a[2:1].bags.bags)
    @test nobs(a[2:1].data) == 0
end

@testset "testing WeightedBagNode indexing" begin
    @test wa[1].data.data == wa.data.data
    @test wa[1].bags.bags == [1:4]
    @test wb[1:2].data.data == wb.data.data
    @test wb[1:2].bags.bags == [1:2,3:4]
    @test wb[2].data.data == wb.data.data[:,3:4]
    @test wb[2].bags.bags == [1:2]
    @test wb[1].data.data == wb.data.data[:,1:2]
    @test wb[1].bags.bags == [1:2]
    @test wc[1].data.data == wc.data.data[:,1:1]
    @test wc[1].bags.bags == [1:1]
    @test wc[[1,2]].data.data == wc.data.data[:,1:2]
    @test wc[[1,2]].bags.bags == [1:1,2:2]
    @test wc[[2,1]].data.data == wc.data.data[:,[2,1]]
    @test wc[[2,1]].bags.bags == [1:1,2:2]
    @test wd[[2,1]].data.data == wd.data.data
    @test wd[[2,1]].bags.bags == [0:-1,1:4]
    @test wd[1:2].data.data == wd.data.data
    @test wd[1:2].bags.bags == [1:4,0:-1]
    @test nobs(wd[2].data) == 0
    @test wd[2].bags.bags == [0:-1]
end

@testset "testing nested ragged array" begin
    x = BagNode(ArrayNode(rand(3,10)),[1:2,3:3,0:-1,4:5,6:6,7:10])
    y = BagNode(x,[1:2,3:3,4:5,6:6])
    @test y[1].data.data.data == x.data.data[:,1:3]
    @test y[1].data.bags.bags == [1:2,3:3]
    @test y[1:2].data.data.data == x.data.data[:,1:3]
    @test y[1:2].data.bags.bags == [1:2,3:3,0:-1]
    @test y[2:3].data.data.data == x.data.data[:,4:6]
    @test y[2:3].data.bags.bags == [0:-1,1:2,3:3]
end


@testset "testing ProductNode" begin
    x = ProductNode((ArrayNode(rand(3,2)),ArrayNode(rand(3,2)),ArrayNode(randn(3,2))))
    y = ProductNode((ArrayNode(rand(3,2)),ArrayNode(rand(3,2)),ArrayNode(randn(3,2))))
    @test catobs(x,y).data[1].data == hcat(x.data[1].data,y.data[1].data)
    @test reduce(catobs, [x,y]).data[1].data == hcat(x.data[1].data,y.data[1].data)
    @test catobs(x,y).data[2].data == hcat(x.data[2].data,y.data[2].data)
    @test reduce(catobs, [x,y]).data[2].data == hcat(x.data[2].data,y.data[2].data)
    @test catobs(x,y).data[3].data == hcat(x.data[3].data,y.data[3].data)
    @test reduce(catobs, [x,y]).data[3].data == hcat(x.data[3].data,y.data[3].data)
    @test cat(x,y, dims = ndims(x)).data[3].data == hcat(x.data[3].data,y.data[3].data)

    @test k[1].data[1].data.data ==  wb[1].data.data
    @test k[2].data[1].data.data ==  wb[2].data.data
    @test l[2:3].data[1].data.data ==  wc[2:3].data.data
end

@testset "testing sparsify" begin
    @test sparsify(zeros(10, 10), 0.05) isa SparseMatrixCSC
    @test sparsify(randn(10, 10), 0.05) isa Matrix
    @test sparsify(randn(10), 0.05) isa Vector
end

@testset "testing sparsify and mapdata" begin
    x = ProductNode((ProductNode((ArrayNode(randn(5,5)), ArrayNode(zeros(5,5)))), ArrayNode(zeros(5,5))))
    xs = mapdata(i -> sparsify(i, 0.05), x)
    @test xs.data[2].data isa SparseMatrixCSC
    @test xs.data[1].data[2].data isa SparseMatrixCSC
    @test xs.data[1].data[1].data == x.data[1].data[1].data
end

@testset "testing missing mapdata" begin
    x = ProductNode((ProductNode((ArrayNode(randn(5,5)), ArrayNode(zeros(5,5)))), ArrayNode(zeros(5,5))), BagNode(missing, AlignedBags([0:-1]), nothing))
    xs = mapdata(i -> sparsify(i, 0.05), x)
    @test xs.data[2].data isa SparseMatrixCSC
    @test xs.data[1].data[2].data isa SparseMatrixCSC
    @test xs.data[1].data[1].data == x.data[1].data[1].data
end

@testset "testing equals and hash" begin
    a2 = deepcopy(a)
    i2 = deepcopy(i)
    k2 = ProductNode((a = wb, b = b))
    metadata1 = fill("metadata", 4)
    metadata2 = "Oh, Hi Mark"
    r = rand(3,4)
    x1 = BagNode(ArrayNode(r),[1:4], metadata1)
    x2 = BagNode(ArrayNode(r),[1:4], metadata1)
    x3 = BagNode(ArrayNode(r),[1:4], metadata2)

    @test a != b
    @test a != i
    @test a == a2
    @test i == i2
    @test i != k
    @test k == k2
    @test x1 == x2
    @test x1 != x3
    @test hash(a) !== hash(b)
    @test hash(a) === hash(a2)
    @test hash(a) !== hash(i)
    @test hash(i) === hash(i2)
    @test hash(i) !== hash(k)
    @test hash(k) === hash(k2)
    @test hash(x1) === hash(x2)
    @test hash(x1) !== hash(x3)
end

@testset "testing equals with missings" begin
    a = ArrayNode([0.0 missing 0.0 0.0 1.0])
    b = ArrayNode([0.0 missing 0.0 0.0 2.0])
    c = ArrayNode([0.0 missing 0.0 missing 2.0])
    @test a != a
    @test isequal(a, a)
    @test a != b
    @test !isequal(a, b)
    @test b != b
    @test isequal(b, b)
    @test a != c
    @test !isequal(a, c)
    @test b != c
    @test !isequal(b, c)
    @test c != c
    @test isequal(c, c)

    d = ProductNode((; a=a, b=b))
    e = ProductNode((; a=a, b=b), [])
    @test isequal(d, d)
    @test !isequal(d, e)
    @test d != d
    @test d != e

    f = BagNode(a,[1:4])
    g = BagNode(b,[1:4])
    @test f != f
    @test f != g
    @test isequal(f, f)
    @test !isequal(f, g)

    wf = WeightedBagNode(a, [1:4], rand(1:4, 4))
    wg = WeightedBagNode(b, [1:4], rand(1:4, 4))
    @test wf != wf
    @test wf != wg
    @test isequal(wf, wf)
    @test !isequal(wf, wg)

    h = LazyNode(:Test, a.data)
    i = LazyNode(:Test, b.data)
    @test isequal(h == h, missing)
    @test h != i
    @test isequal(h, h)
    @test !isequal(h, i)
end
