using Mill
using SparseArrays, DataFrames

import Mill: sparsify, mapdata
import LearnBase: nobs

@testset "creating bags" begin
    k = [2, 2, 2, 1, 1, 3]
    @test Mill.bags(k).bags == [1:3,4:5,6:6]
end

metadata = fill("metadata", 4)
a = BagNode(ArrayNode(rand(3,4)),[1:4], metadata)
b = BagNode(ArrayNode(rand(3,4)),[1:2,3:4], metadata)
c = BagNode(ArrayNode(rand(3,4)),[1:1,2:2,3:4], metadata)
d = BagNode(ArrayNode(rand(3,4)),[1:4,0:-1], metadata)
wa = WeightedBagNode(ArrayNode(rand(3,4)),[1:4], rand(1:4, 4), metadata)
wb = WeightedBagNode(ArrayNode(rand(3,4)),[1:2,3:4], rand(1:4, 4), metadata)
wc = WeightedBagNode(ArrayNode(rand(3,4)),[1:1,2:2,3:4], rand(1:4, 4), metadata)
wd = WeightedBagNode(ArrayNode(rand(3,4)),[1:4,0:-1], rand(1:4, 4), metadata)
e = ArrayNode(rand(2, 2))

f = TreeNode((wb,b))
g = TreeNode([c, wc])
h = TreeNode((wc,c))
i = TreeNode((
              b,
              TreeNode((
                        b,
                        BagNode(
                                BagNode(
                                        ArrayNode(rand(2,4)),
                                        [1:1, 2:2, 3:3, 4:4]
                                       ),
                                [1:3, 4:4]
                               )
                       ))
             ))
k = TreeNode((a = wb, b = b))
l = TreeNode((a = wc, b = c))

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
    @test all(catobs(e, e).data .== hcat(e.data, e.data) .== reduce(catobs, [e,e]).data)
    @test all(hcat(e, e).data .== hcat(e.data, e.data))
    @test all(vcat(e, e).data .== vcat(e.data, e.data))
    x = ArrayNode(randn(2,3),rand(3))
    @test typeof(catobs(x,x[0:-1])) .== ArrayNode{Array{Float64,2},Array{Float64,1}}
    @inferred catobs(x,x[0:-1]) == [1,2,3]
    @test typeof(reduce(catobs, [x, x[0:-1]])) .== ArrayNode{Array{Float64,2},Array{Float64,1}}
    @inferred reduce(catobs, [x, x[0:-1]]) == [1,2,3]
    @test all(cat(e, e, dims = ndims(e)).data .== hcat(e.data, e.data))
end

@testset "testing BagNode hcat" begin
    @test all(catobs(a, b, c).data.data .== hcat(a.data.data, b.data.data, c.data.data))
    @test all(reduce(catobs, [a, b, c]).data.data .== hcat(a.data.data, b.data.data, c.data.data))
    @test catobs(a, b, c).bags.bags == vcat(a.bags, b.bags, c.bags).bags
    @test reduce(catobs, [a, b, c]).bags.bags == vcat(a.bags, b.bags, c.bags).bags
    @test all(catobs(c, a).data.data .== hcat(c.data.data, a.data.data))
    @test all(reduce(catobs, [c, a]).data.data .== hcat(c.data.data, a.data.data))
    @test catobs(c, a).bags.bags == vcat(c.bags, a.bags).bags
    @test reduce(catobs, [c, a]).bags.bags == vcat(c.bags, a.bags).bags
    @test all(catobs(a, c).data.data .== hcat(a.data.data, c.data.data))
    @test all(reduce(catobs, [a, c]).data.data .== hcat(a.data.data, c.data.data))
    @test catobs(a, c).bags.bags == vcat(a.bags, c.bags).bags
    @test reduce(catobs, [a, c]).bags.bags == vcat(a.bags, c.bags).bags
    @test all(catobs(a, d).data.data .== hcat(a.data.data, d.data.data))
    @test all(reduce(catobs, [a, d]).data.data .== hcat(a.data.data, d.data.data))
    @test catobs(a, d).bags.bags == vcat(a.bags, d.bags).bags
    @test reduce(catobs, [a, d]).bags.bags == vcat(a.bags, d.bags).bags
    @test all(catobs(d, a).data.data .== hcat(d.data.data, a.data.data))
    @test all(reduce(catobs, [d, a]).data.data .== hcat(d.data.data, a.data.data))
    @test catobs(d, a).bags.bags == vcat(d.bags, a.bags).bags
    @test reduce(catobs, [d, a]).bags.bags == vcat(d.bags, a.bags).bags
    @test all(catobs(d).data.data .== hcat(d.data.data))
    @test all(reduce(catobs, [d]).data.data .== hcat(d.data.data))
    @test catobs(d).bags.bags == vcat(d.bags).bags
    @test reduce(catobs, [d]).bags.bags == vcat(d.bags).bags
    @test all(cat(a, b, dims = ndims(a)).data.data .== hcat(a.data.data, b.data.data))
end

@testset "testing WeightedBagNode hcat" begin
    @test all(catobs(wa, wb, wc).data.data .== hcat(wa.data.data, wb.data.data, wc.data.data))
    @test all(reduce(catobs, [wa, wb, wc]).data.data .== hcat(wa.data.data, wb.data.data, wc.data.data))
    @test catobs(wa, wb, wc).bags.bags == vcat(wa.bags, wb.bags, wc.bags).bags
    @test reduce(catobs, [wa, wb, wc]).bags.bags == vcat(wa.bags, wb.bags, wc.bags).bags
    @test all(catobs(wc, wa).data.data .== hcat(wc.data.data, wa.data.data))
    @test all(reduce(catobs, [wc, wa]).data.data .== hcat(wc.data.data, wa.data.data))
    @test catobs(wc, wa).bags.bags == vcat(wc.bags, wa.bags).bags
    @test reduce(catobs, [wc, wa]).bags.bags == vcat(wc.bags, wa.bags).bags
    @test all(catobs(wa, wc).data.data .== hcat(wa.data.data, wc.data.data))
    @test all(reduce(catobs, [wa, wc]).data.data .== hcat(wa.data.data, wc.data.data))
    @test catobs(wa, wc).bags.bags == vcat(wa.bags, wc.bags).bags
    @test reduce(catobs, [wa, wc]).bags.bags == vcat(wa.bags, wc.bags).bags
    @test all(catobs(wa, wd).data.data .== hcat(wa.data.data, wd.data.data))
    @test all(reduce(catobs, [wa, wd]).data.data .== hcat(wa.data.data, wd.data.data))
    @test catobs(wa, wd).bags.bags == vcat(wa.bags, wd.bags).bags
    @test reduce(catobs, [wa, wd]).bags.bags == vcat(wa.bags, wd.bags).bags
    @test all(catobs(wd, wa).data.data .== hcat(wd.data.data, wa.data.data))
    @test all(reduce(catobs, [wd, wa]).data.data .== hcat(wd.data.data, wa.data.data))
    @test catobs(wd, wa).bags.bags == vcat(wd.bags, wa.bags).bags
    @test reduce(catobs, [wd, wa]).bags.bags == vcat(wd.bags, wa.bags).bags
    @test all(catobs(wd).data.data .== hcat(wd.data.data))
    @test all(reduce(catobs, [wd]).data.data .== hcat(wd.data.data))
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

@testset "testing hierarchical hcat on tree nodes" begin
    @test all(catobs(f, h).data[1].data.data .== hcat(wb.data.data, wc.data.data))
    @test all(reduce(catobs, [f, h]).data[1].data.data .== hcat(wb.data.data, wc.data.data))
    @test all(catobs(f, h).data[2].data.data .== hcat(b.data.data, c.data.data))
    @test all(reduce(catobs, [f, h]).data[2].data.data .== hcat(b.data.data, c.data.data))
    @test all(catobs(f, h, f).data[1].data.data .== hcat(wb.data.data, wc.data.data, wb.data.data))
    @test all(reduce(catobs, [f, h, f]).data[1].data.data .== hcat(wb.data.data, wc.data.data, wb.data.data))

    @test all(catobs(k, l).data[1].data.data .== hcat(wb.data.data, wc.data.data))
    @test all(catobs(k, l).data[2].data.data .== hcat(b.data.data, c.data.data))
    @test nobs(catobs(k,l)) == nobs(k) + nobs(l)

    # different tuple length
    @test_skip @test_throws Exception catobs(f, i)
    @test_skip @test_throws Exception reduce(catobs, [f, i])
    @test_skip @test_throws Exception catobs(g, i)
    @test_skip @test_throws Exception reduce(catobs, [g, i])
    # different content
    @test_skip @test_throws Exception catobs(f, g)
    @test_skip @test_throws Exception reduce(catobs, [f, g])
end

@testset "testing BagNode indexing" begin
    @test all(a[1].data.data .== a.data.data)
    @test a[1].bags.bags == [1:4]
    @test all(b[1:2].data.data .== b.data.data)
    @test b[1:2].bags.bags == [1:2,3:4]
    @test all(b[2].data.data .== b.data.data[:,3:4])
    @test b[2].bags.bags == [1:2]
    @test all(b[1].data.data .== b.data.data[:,1:2])
    @test b[1].bags.bags == [1:2]
    @test all(c[1].data.data .== c.data.data[:,1:1])
    @test c[1].bags.bags == [1:1]
    @test all(c[[1,2]].data.data .== c.data.data[:,1:2])
    @test c[[1,2]].bags.bags == [1:1,2:2]
    @test all(c[[2,1]].data.data .== c.data.data[:,[2,1]])
    @test c[[2,1]].bags.bags == [1:1,2:2]
    @test all(d[[2,1]].data.data .== d.data.data)
    @test d[[2,1]].bags.bags == [0:-1,1:4]
    @test all(d[1:2].data.data .== d.data.data)
    @test d[1:2].bags.bags == [1:4,0:-1]
    @test ismissing(d[2].data)
    @test d[2].bags.bags == [0:-1]
    @test isempty(a[2:1].bags.bags)
    @test ismissing(a[2:1].data)
end

@testset "testing WeightedBagNode indexing" begin
    @test all(wa[1].data.data .== wa.data.data)
    @test wa[1].bags.bags == [1:4]
    @test all(wb[1:2].data.data .== wb.data.data)
    @test wb[1:2].bags.bags == [1:2,3:4]
    @test all(wb[2].data.data .== wb.data.data[:,3:4])
    @test wb[2].bags.bags == [1:2]
    @test all(wb[1].data.data .== wb.data.data[:,1:2])
    @test wb[1].bags.bags == [1:2]
    @test all(wc[1].data.data .== wc.data.data[:,1:1])
    @test wc[1].bags.bags == [1:1]
    @test all(wc[[1,2]].data.data .== wc.data.data[:,1:2])
    @test wc[[1,2]].bags.bags == [1:1,2:2]
    @test all(wc[[2,1]].data.data .== wc.data.data[:,[2,1]])
    @test wc[[2,1]].bags.bags == [1:1,2:2]
    @test all(wd[[2,1]].data.data .== wd.data.data)
    @test wd[[2,1]].bags.bags == [0:-1,1:4]
    @test all(wd[1:2].data.data .== wd.data.data)
    @test wd[1:2].bags.bags == [1:4,0:-1]
    @test ismissing(wd[2].data)
    @test wd[2].bags.bags == [0:-1]
end

@testset "testing nested ragged array" begin
    x = BagNode(ArrayNode(rand(3,10)),[1:2,3:3,0:-1,4:5,6:6,7:10])
    y = BagNode(x,[1:2,3:3,4:5,6:6])
    @test all(y[1].data.data.data .== x.data.data[:,1:3])
    @test y[1].data.bags.bags == [1:2,3:3]
    @test all(y[1:2].data.data.data .== x.data.data[:,1:3])
    @test y[1:2].data.bags.bags == [1:2,3:3,0:-1]
    @test all(y[2:3].data.data.data .== x.data.data[:,4:6])
    @test y[2:3].data.bags.bags == [0:-1,1:2,3:3]
end


@testset "testing TreeNode" begin
    x = TreeNode((ArrayNode(rand(3,2)),ArrayNode(rand(3,2)),ArrayNode(randn(3,2))))
    y = TreeNode((ArrayNode(rand(3,2)),ArrayNode(rand(3,2)),ArrayNode(randn(3,2))))
    @test all(catobs(x,y).data[1].data .== hcat(x.data[1].data,y.data[1].data))
    @test all(reduce(catobs, [x,y]).data[1].data .== hcat(x.data[1].data,y.data[1].data))
    @test all(catobs(x,y).data[2].data .== hcat(x.data[2].data,y.data[2].data))
    @test all(reduce(catobs, [x,y]).data[2].data .== hcat(x.data[2].data,y.data[2].data))
    @test all(catobs(x,y).data[3].data .== hcat(x.data[3].data,y.data[3].data))
    @test all(reduce(catobs, [x,y]).data[3].data .== hcat(x.data[3].data,y.data[3].data))
    @test all(cat(x,y, dims = ndims(x)).data[3].data .== hcat(x.data[3].data,y.data[3].data))

    @test all(k[1].data[1].data.data .==  wb[1].data.data)
    @test all(k[2].data[1].data.data .==  wb[2].data.data)
    @test all(l[2:3].data[1].data.data .==  wc[2:3].data.data)
end

@testset "testing sparsify" begin
    @test typeof(sparsify(zeros(10,10),0.05)) <: SparseMatrixCSC
    @test typeof(sparsify(randn(10,10),0.05)) <: Matrix
    @test typeof(sparsify(randn(10),0.05)) <: Vector
end

@testset "testing sparsify and mapdata" begin
    x = TreeNode((TreeNode((ArrayNode(randn(5,5)), ArrayNode(zeros(5,5)))), ArrayNode(zeros(5,5))))
    xs = mapdata(i -> sparsify(i, 0.05), x)
    @test typeof(xs.data[2].data) <: SparseMatrixCSC
    @test typeof(xs.data[1].data[2].data) <: SparseMatrixCSC
    @test all(xs.data[1].data[1].data .== x.data[1].data[1].data)
end

@testset "testing missing mapdata" begin
    x = TreeNode((TreeNode((ArrayNode(randn(5,5)), ArrayNode(zeros(5,5)))), ArrayNode(zeros(5,5))), BagNode(missing, AlignedBags([0:-1]), nothing))
    xs = mapdata(i -> sparsify(i, 0.05), x)
    @test typeof(xs.data[2].data) <: SparseMatrixCSC
    @test typeof(xs.data[1].data[2].data) <: SparseMatrixCSC
    @test all(xs.data[1].data[1].data .== x.data[1].data[1].data)
end
