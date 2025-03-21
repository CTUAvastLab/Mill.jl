md1 = ["metadata"]
md2 = fill("metadata", 2)
md3 = fill("metadata", 3)
a = BagNode(rand(3, 4), [1:4], md1)
b = BagNode(rand(3, 4), [1:2, 3:4], md2)
c = BagNode(rand(3, 4), [1:1, 2:2, 3:4], md3)
d = BagNode(rand(3, 4), [1:4, 0:-1], md2)
wa = WeightedBagNode(rand(3, 4), [1:4], rand(1:4, 4) .|> Float64, md1)
wb = WeightedBagNode(rand(3, 4), [1:2, 3:4], rand(1:4, 4) .|> Float64, md2)
wc = WeightedBagNode(rand(3, 4), [1:1, 2:2, 3:4], rand(1:4, 4) .|> Float64, md3)
wd = WeightedBagNode(rand(3, 4), [1:4, 0:-1], rand(1:4, 4) .|> Float64, md2)
e = ArrayNode(rand(2, 2), md2)

f = ProductNode((wb,b), md2)
g = ProductNode([c, wc], md3)
h = ProductNode((wc,c), md3)
i = ProductNode((
              b,
              ProductNode((
                        b,
                        BagNode(
                                BagNode(
                                        rand(2, 4),
                                        [1:1, 2:2, 3:3, 4:4]
                                       ),
                                [1:3, 4:4]
                               )
                       ))
             ), md2)
k = ProductNode((a = wb, b = b), md2)
l = ProductNode((a = wc, b = c), md3)
m = ProductNode((a = wc, c = c), md3)
n = ProductNode((a = c, c = wc), md3)

@testset "constructor logic" begin
    x = randn(2, 2)
    n1 = ArrayNode(x)
    bs = [1:1, 2:2]
    b = bags(bs)
    w = [0.1, 0.2]

    for md in [tuple(), tuple(nothing), tuple("metadata")]
        n2 = BagNode(n1, b, md...)
        @test n2 isa BagNode{typeof(n1), typeof(b), isempty(md) ? Nothing : typeof(only(md))}
        @test n2 == BagNode(x, b, md...) == BagNode(n1, bs, md...) == BagNode(x, bs, md...)

        n2 = WeightedBagNode(n1, b, w, md...)
        @test n2 isa WeightedBagNode{typeof(n1), typeof(b), eltype(w),
                isempty(md) ? Nothing : typeof(only(md))}
        @test n2 == WeightedBagNode(x, b, w, md...) == WeightedBagNode(n1, bs, w, md...) ==
            WeightedBagNode(x, bs, w, md...)

        n3 = ProductNode(tuple(n2), md...)
        @test n3 isa ProductNode{Tuple{typeof(n2)}, isempty(md) ? Nothing : typeof(only(md))}
        @test n3 == ProductNode(n2, md...)

        n4 = ProductNode((n1, n1), md...)
        @test n4 isa ProductNode{Tuple{typeof(n1), typeof(n1)},
                isempty(md) ? Nothing : typeof(only(md))}
        @test n4 == ProductNode((x, x), md...) == ProductNode((n1, x), md...) ==
                    ProductNode((x, n1), md...)

        n5 = ProductNode((a=n1, b=n2), md...)
        @test n5 isa ProductNode{NamedTuple{(:a, :b), Tuple{typeof(n1), typeof(n2)}},
                isempty(md) ? Nothing : typeof(only(md))}
        @test n5 == ProductNode((a=x, b=n2), md...) ==
                    ProductNode(md...; a=n1, b=n2) ==
                    ProductNode(md...; a=x, b=n2)

        n6 = LazyNode{:Test}(n1, md...)
        @test n6 isa LazyNode{:Test, typeof(n1), isempty(md) ? Nothing : typeof(only(md))}
        @test n6 == LazyNode(:Test, n1, md...)
    end
end

@testset "constructor assertions" begin
    @test_throws AssertionError BagNode(missing, length2bags([0, 1, 0]))
    @test_throws AssertionError BagNode(e, length2bags([1, 1, 1]))
    @test_throws AssertionError BagNode(i, ScatteredBags([[1, 2], [3]]))

    for (x, y) in [(g, i), (f, g), (k, m)]
        @test_throws AssertionError ProductNode((x, y))
        @test_throws AssertionError ProductNode(; x, y)
        @test_throws AssertionError ProductNode((x, x, y))
    end
end

@testset "numobs" begin
    @test numobs(a) == numobs(wa) == 1
    @test numobs(b) == numobs(wb) == 2
    @test numobs(c) == numobs(wc) == 3
    @test numobs(d) == numobs(wd) == 2
    @test numobs(e) == 2
    @test numobs(f) == numobs(wb) == numobs(b) == numobs(k)
    @test numobs(g) == numobs(c) == numobs(wc) == numobs(l)
    @test numobs(h) == numobs(wc) == numobs(c)
    @test numobs(i) == numobs(b)
end

@testset "ArrayNode catobs, hcat and vcat" begin
    @test hcat(e, e).data == hcat(e.data, e.data)
    @test hcat(e, e).metadata == hcat(e.metadata, e.metadata)
    @test_nowarn @inferred hcat(e, e)

    @test vcat(e, e).data == vcat(e.data, e.data)
    @test vcat(e, e).metadata == vcat(e.metadata, e.metadata)
    @test_nowarn @inferred vcat(e, e)

    @test hcat(e, e).data == catobs(e, e).data
    @test vcat(e, e).metadata == catobs(e, e).metadata

    @test catobs(e, e) == reduce(catobs, [e, e]) == cat(e, e, dims=ndims(e))
    @test_nowarn @inferred catobs(e, e)
    @test_nowarn @inferred reduce(catobs, [e, e])
    @test_nowarn @inferred cat(e, e, dims=ndims(e))

    x = ArrayNode(randn(2, 3), rand(2, 3))
    @test catobs(x, x[0:-1]) isa ArrayNode{Matrix{Float64}, Matrix{Float64}}
    @test_nowarn @inferred catobs(x, x[0:-1])
    @test reduce(catobs, [x, x[0:-1]]) isa ArrayNode{Matrix{Float64}, Matrix{Float64}}
    @test_nowarn @inferred reduce(catobs, [x, x[0:-1]])
end

@testset "BagNode catobs" begin
    for ns in [(a, b, c), (c, a), (a, d), (d, a), (d,)]
        @test catobs(ns...).data == hcat(map(Mill.data, ns)...)
        @test catobs(ns...).bags == vcat(map(n -> n.bags, ns)...)
        @test catobs(ns...).metadata == vcat(map(Mill.metadata, ns)...)
        @test reduce(catobs, ns) == catobs(ns...) == cat(ns..., dims=ndims(ns[1]))
    end
end

@testset "WeighteBagNode catobs" begin
    for ns in [(wa, wb, wc), (wc, wa), (wa, wd), (wd, wa), (wd,)]
        @test catobs(ns...).data == hcat(map(Mill.data, ns)...)
        @test catobs(ns...).weights == vcat(map(n -> n.weights, ns)...)
        @test catobs(ns...).bags == vcat(map(n -> n.bags, ns)...)
        @test catobs(ns...).metadata == vcat(map(Mill.metadata, ns)...)
        @test reduce(catobs, ns) == catobs(ns...) == cat(ns..., dims=ndims(ns[1]))
    end
end

@testset "ProductNodes catobs" begin
    @test catobs(f, h) == reduce(catobs, [f, h])
    @test catobs(f, h, f) == reduce(catobs, [f, h, f])
    @test catobs(g, g) == reduce(catobs, [g, g])
    @test catobs(k, l) == reduce(catobs, [k, l])

    @test catobs(f, h).data[1].data.data == hcat(wb.data.data, wc.data.data)
    @test catobs(f, h).data[2].data.data == hcat(b.data.data, c.data.data)
    @test catobs(f, h, f).data[1].data.data == hcat(wb.data.data, wc.data.data, wb.data.data)
    @test numobs(catobs(f, h)) == numobs(f) + numobs(h)

    @test catobs(g, g).data[1].data.data == hcat(c.data.data, c.data.data)
    @test catobs(g, g).data[2].data.data == hcat(wc.data.data, wc.data.data)
    @test catobs(g, g, g).data[1].data.data == hcat(c.data.data, c.data.data, c.data.data)
    @test numobs(catobs(g, g)) == 2numobs(g)

    @test catobs(k, l).data[1].data.data == hcat(wb.data.data, wc.data.data)
    @test catobs(k, l).data[2].data.data == hcat(b.data.data, c.data.data)
    @test numobs(catobs(k,l)) == numobs(k) + numobs(l)

    # correct length/keyset but different subtrees
    @test_throws MethodError catobs(f, i)
    @test_throws MethodError reduce(catobs, [f, i])
    @test_throws MethodError catobs(m, n)
    @test_throws MethodError reduce(catobs, [m, n])
    # different tuple length or keyset
    @test_throws ArgumentError catobs(g, i)
    @test_throws ArgumentError reduce(catobs, [g, i])
    @test_throws ArgumentError catobs(f, g)
    @test_throws ArgumentError reduce(catobs, [f, g])
    @test_throws ArgumentError catobs(k, m)
    @test_throws ArgumentError reduce(catobs, [k, m])
    @test_throws ArgumentError catobs(l, m)
    @test_throws ArgumentError reduce(catobs, [l, m])
end

@testset "catobs with missing nodes" begin
    @test catobs(e, e) == catobs(e, missing, e) == catobs(missing, e, missing, e)
    @test catobs(a, b, c) == catobs(a, missing, b, c) == catobs(missing, a, b, c)
    @test catobs(wa, wb, wc) == catobs(wa, wb, missing, wc) == catobs(wa, wb, wc, missing)
    @test catobs(k, l) == catobs(k, missing, l) == catobs(missing, k, l)
end

@testset "catobs with missing data" begin
    @test catobs(a, b) == catobs(a, BagNode(missing, AlignedBags()), b)
    @test catobs(wa, wb) == catobs(WeightedBagNode(missing, AlignedBags(), Float64[]), wa, wb)
end

@testset "catobs with `nothing` metadata" begin
    ee = catobs(e, ArrayNode(zeros(2, 0)))
    @test e.metadata == ee.metadata

    aa = catobs(a, BagNode(zeros(3, 0), AlignedBags()))
    @test a.metadata == aa.metadata
    @test a.data.metadata == aa.data.metadata

    waa = catobs(wa, WeightedBagNode(zeros(3, 0), AlignedBags(), zeros(0)))
    @test wa.metadata == waa.metadata
    @test wa.data.metadata == waa.data.metadata

    ff = catobs(f, ProductNode((
        WeightedBagNode(zeros(3, 0), AlignedBags(), zeros(0)),
        BagNode(zeros(3, 0), AlignedBags())
    )))
    @test f.metadata == ff.metadata
    @test f.data[1].metadata == ff.data[1].metadata
    @test f.data[2].metadata == ff.data[2].metadata
    @test f.data[1].data.metadata == ff.data[1].data.metadata
    @test f.data[2].data.metadata == ff.data[2].data.metadata
end

@testset "catobs stability" begin
    for n in [a, b, c, d, e, f, h, k, l, wa, wb, wc, wd]
        @test_nowarn @inferred catobs(n, n)
        @test_nowarn @inferred reduce(catobs, [n, n])
        @test_nowarn @inferred catobs([n, n])
        @test_nowarn @inferred cat(n, n, dims=ndims(n))
    end
end

@testset "MLUtils.batch and MLUtils.unbatch" begin
    a = ArrayNode(rand(2, 1))
    b = ArrayNode(rand(2, 1))
    @test [a, b] == MLUtils.unbatch(MLUtils.batch([a, b]))
    a = BagNode(rand(2, 1), [1:1])
    b = BagNode(rand(2, 0), [0:-1])
    @test [a, b] == MLUtils.unbatch(MLUtils.batch([a, b]))
    wa = WeightedBagNode(rand(2, 1), [1:1], rand(1))
    wb = WeightedBagNode(rand(2, 0), [0:-1], Float64[])
    @test [wa, wb] == MLUtils.unbatch(MLUtils.batch([wa, wb]))
    a = ProductNode((a, wa))
    b = ProductNode((b, wb))
    @test [a, b] == MLUtils.unbatch(MLUtils.batch([a, b]))
end

@testset "BagNode indexing" begin
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
    @test numobs(d[2].data) == 0
    @test d[2].bags.bags == [0:-1]
    @test isempty(a[2:1].bags.bags)
    @test numobs(a[2:1].data) == 0
end

@testset "WeightedBagNode indexing" begin
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
    @test numobs(wd[2].data) == 0
    @test wd[2].bags.bags == [0:-1]
end

@testset "ProductNode indexing" begin
    @test h[1].data[1].data.data ==  wc[1].data.data
    @test h[[3,2]].data[1].data.data ==  wc[[3,2]].data.data
    @test h[1:2].data[2].data.data ==  c[1:2].data.data
    @test h[end].data[2].data.data ==  c[end].data.data
    @test g[1].data[1].data.data ==  c[1].data.data
    @test g[[3,2]].data[1].data.data ==  c[[3,2]].data.data
    @test g[1:2].data[2].data.data ==  wc[1:2].data.data
    @test g[end].data[2].data.data ==  wc[end].data.data
    @test l[1].data[:a].data.data ==  wc[1].data.data
    @test l[[3,2]].data[:a].data.data ==  wc[[3,2]].data.data
    @test l[1:2].data[:b].data.data ==  c[1:2].data.data
    @test l[end].data[:b].data.data ==  c[end].data.data
end

@testset "nested ragged array" begin
    x = BagNode(rand(3,10),[1:2,3:3,0:-1,4:5,6:6,7:10])
    y = BagNode(x, [1:2,3:3,4:5,6:6])
    @test y[1].data.data.data == x.data.data[:,1:3]
    @test y[1].data.bags.bags == [1:2,3:3]
    @test y[1:2].data.data.data == x.data.data[:,1:3]
    @test y[1:2].data.bags.bags == [1:2,3:3,0:-1]
    @test y[2:3].data.data.data == x.data.data[:,4:6]
    @test y[2:3].data.bags.bags == [0:-1,1:2,3:3]
end

@testset "keys and haskey" begin
    @test keys(g) == [1, 2]
    @test keys(h) == [1, 2]
    @test keys(k) == (:a, :b)

    @test haskey(k, :a)
    @test haskey(k, :b)
    @test !haskey(k, :c)
end

@testset "equals and hash" begin
    a2 = deepcopy(a)
    i2 = deepcopy(i)
    k2 = ProductNode((a = wb, b), md2)
    metadata1 = fill("metadata", 4)
    metadata2 = "Oh, Hi Mark"
    r = rand(3,4)
    x1 = BagNode(r, [1:4], metadata1)
    x2 = BagNode(r, [1:4], metadata1)
    x3 = BagNode(r, [1:4], metadata2)

    @test a ≠ b
    @test a ≠ i
    @test a == a2
    @test i == i2
    @test i ≠ k
    @test k == k2
    @test x1 == x2
    @test x1 ≠ x3
    @test hash(a) ≢ hash(b)
    @test hash(a) ≡ hash(a2)
    @test hash(a) ≢ hash(i)
    @test hash(i) ≡ hash(i2)
    @test hash(i) ≢ hash(k)
    @test hash(k) ≡ hash(k2)
    @test hash(x1) ≡ hash(x2)
    @test hash(x1) ≢ hash(x3)
end

@testset "equals with missings" begin
    a = ArrayNode([0.0 missing 0.0 0.0 1.0])
    b = ArrayNode([0.0 missing 0.0 0.0 2.0])
    c = ArrayNode([0.0 missing 0.0 missing 2.0])
    @test a ≠ a
    @test isequal(a, a)
    @test a ≠ b
    @test !isequal(a, b)
    @test b ≠ b
    @test isequal(b, b)
    @test a ≠ c
    @test !isequal(a, c)
    @test b ≠ c
    @test !isequal(b, c)
    @test c ≠ c
    @test isequal(c, c)

    d = ProductNode(; a, b)
    e = ProductNode((; a, b), [])
    @test isequal(d, d)
    @test !isequal(d, e)
    @test d ≠ d
    @test d ≠ e

    f = BagNode(a,[1:4])
    g = BagNode(b,[1:4])
    @test f ≠ f
    @test f ≠ g
    @test isequal(f, f)
    @test !isequal(f, g)

    wf = WeightedBagNode(a, [1:4], rand(1:4, 4))
    wg = WeightedBagNode(b, [1:4], rand(1:4, 4))
    @test wf ≠ wf
    @test wf ≠ wg
    @test isequal(wf, wf)
    @test !isequal(wf, wg)

    h = LazyNode(:Test, a.data)
    i = LazyNode(:Test, b.data)
    @test isequal(h == h, missing)
    @test h ≠ i
    @test isequal(h, h)
    @test !isequal(h, i)
end

@testset "dropmeta" begin
    for n in [a, b, c, d, e, f, h, k, l, wa, wb, wc, wd]
        @test NodeIterator(dropmeta(n)) |> collect .|> Mill.metadata .|> isnothing |> all
    end
end
