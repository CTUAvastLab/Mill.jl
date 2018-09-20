using Mill: ArrayNode, BagNode, TreeNode, WeightedBagNode
using SparseArrays

import Mill: sparsify, mapdata
import LearnBase: nobs
let 

    @testset "creating bags" begin
        k = [2, 2, 2, 1, 1, 3]
        @test all(Mill.bag(k) .== [1:3,4:5,6:6])
    end

    @testset "testing remapping for subsets" begin
        @test all(Mill.remapbag([1:1,2:3,4:5],[2,3])[1] .== [1:2,3:4])
        @test all(Mill.remapbag([1:1,2:3,4:5],[2,3])[2] .== [2,3,4,5])
        @test all(Mill.remapbag([1:2,3:3,4:5],[1,3])[1] .== [1:2,3:4])
        @test all(Mill.remapbag([1:2,3:3,4:5],[1,3])[2] .== [1,2,4,5])
        @test all(Mill.remapbag([1:2,3:3,4:5],[2,3])[1] .== [1:1,2:3])
        @test all(Mill.remapbag([1:2,3:3,4:5],[2,3])[2] .== [3,4,5])

        @test all(Mill.remapbag([1:2,0:-1,3:4],[2,3])[1] .== [0:-1,1:2])
        @test all(Mill.remapbag([1:2,0:-1,3:4],[2,3])[2] .== [3,4])
    end

    a = BagNode(ArrayNode(rand(3,4)),[1:4], ["metadata", "metadata", "metadata", "metadata"])
    b = BagNode(ArrayNode(rand(3,4)),[1:2,3:4])
    c = BagNode(ArrayNode(rand(3,4)),[1:1,2:2,3:4], ["metadata", "metadata", "metadata", "metadata"])
    d = BagNode(ArrayNode(rand(3,4)),[1:4,0:-1])
    wa = WeightedBagNode(ArrayNode(rand(3,4)),[1:4], rand(1:4, 4))
    wb = WeightedBagNode(ArrayNode(rand(3,4)),[1:2,3:4], rand(1:4, 4), ["metadata", "metadata", "metadata", "metadata"])
    wc = WeightedBagNode(ArrayNode(rand(3,4)),[1:1,2:2,3:4], rand(1:4, 4))
    wd = WeightedBagNode(ArrayNode(rand(3,4)),[1:4,0:-1], rand(1:4, 4), ["metadata", "metadata", "metadata", "metadata"])
    e = ArrayNode(rand(2, 2))

    f = TreeNode((wb,b))
    g = TreeNode((c,wc))
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

    @testset "testing nobs" begin
        @test nobs(a) == nobs(wa) == 1
        @test nobs(b) == nobs(wb) == 2
        @test nobs(c) == nobs(wc) == 3
        @test nobs(d) == nobs(wd) == 2
        @test nobs(e) == 2
        @test nobs(f) == nobs(wb) == nobs(b)
        @test nobs(g) == nobs(c) == nobs(wc)
        @test nobs(h) == nobs(wc) == nobs(c)
        @test nobs(i) == nobs(b)
    end

    @testset "testing ArrayNode hcat and vcat" begin
        @test all(cat(e, e).data .== hcat(e.data, e.data))
        @test all(hcat(e, e).data .== hcat(e.data, e.data))
        @test all(vcat(e, e).data .== vcat(e.data, e.data))
        x = ArrayNode(randn(2,3),rand(3))
        @test typeof(cat(x,x[0:-1])) .== ArrayNode{Array{Float64,2},Array{Float64,1}}
    end

    @testset "testing BagNode hcat" begin
        @test all(cat(a, b, c).data.data .== hcat(a.data.data, b.data.data, c.data.data))
        @test all(cat(a, b, c).bags .== [1:4, 5:6, 7:8, 9:9, 10:10, 11:12])
        @test all(cat(c, a).data.data .== hcat(c.data.data, a.data.data))
        @test all(cat(c, a).bags .== [1:1, 2:2, 3:4, 5:8])
        @test all(cat(a, c).data.data .== hcat(a.data.data, c.data.data))
        @test all(cat(a, c).bags .== [1:4, 5:5, 6:6, 7:8])
        @test all(cat(a, d).data.data .== hcat(a.data.data, d.data.data))
        @test all(cat(a, d).bags .== [1:4, 5:8, 0:-1])
        @test all(cat(d, a).data.data .== hcat(d.data.data, a.data.data))
        @test all(cat(d, a).bags .== [1:4, 0:-1, 5:8])
        @test all(cat(d).data.data .== hcat(d.data.data))
        @test all(cat(d).bags .== [1:4, 0:-1])
    end

    @testset "testing WeightedBagNode hcat" begin
        @test all(cat(wa, wb, wc).data.data .== hcat(wa.data.data, wb.data.data, wc.data.data))
        @test all(cat(wa, wb, wc).bags .== [1:4, 5:6, 7:8, 9:9, 10:10, 11:12])
        @test all(cat(wc, wa).data.data .== hcat(wc.data.data, wa.data.data))
        @test all(cat(wc, wa).bags .== [1:1, 2:2, 3:4, 5:8])
        @test all(cat(wa, wc).data.data .== hcat(wa.data.data, wc.data.data))
        @test all(cat(wa, wc).bags .== [1:4, 5:5, 6:6, 7:8])
        @test all(cat(wa, wd).data.data .== hcat(wa.data.data, wd.data.data))
        @test all(cat(wa, wd).bags .== [1:4, 5:8, 0:-1])
        @test all(cat(wd, wa).data.data .== hcat(wd.data.data, wa.data.data))
        @test all(cat(wd, wa).bags .== [1:4, 0:-1, 5:8])
        @test all(cat(wd).data.data .== hcat(wd.data.data))
        @test all(cat(wd).bags .== [1:4, 0:-1])
    end


    @testset "testing hierarchical hcat on tree nodes" begin
        @test all(cat(f, h).data[1].data.data .== hcat(wb.data.data, wc.data.data))
        @test all(cat(f, h).data[2].data.data .== hcat(b.data.data, c.data.data))
        @test all(cat(f, h, f).data[1].data.data .== hcat(wb.data.data, wc.data.data, wb.data.data))

        # different tuple length
        @test_throws Exception cat(f, i)
        @test_throws Exception cat(g, i)
        # different content
        @test_throws Exception cat(f, g)
    end

    @testset "testing BagNode indexing" begin
        @test all(a[1].data.data .== a.data.data)
        @test all(a[1].bags .== [1:4])
        @test all(b[1:2].data.data .== b.data.data)
        @test all(b[1:2].bags .== [1:2,3:4])
        @test all(b[2].data.data .== b.data.data[:,3:4])
        @test all(b[2].bags .== [1:2])
        @test all(b[1].data.data .== b.data.data[:,1:2])
        @test all(b[1].bags .== [1:2])
        @test all(c[1].data.data .== c.data.data[:,1:1])
        @test all(c[1].bags .== [1:1])
        @test all(c[[1,2]].data.data .== c.data.data[:,1:2])
        @test all(c[[1,2]].bags .== [1:1,2:2])
        @test all(c[[2,1]].data.data .== c.data.data[:,[2,1]])
        @test all(c[[2,1]].bags .== [1:1,2:2])
        @test all(d[[2,1]].data.data .== d.data.data)
        @test all(d[[2,1]].bags .== [0:-1,1:4])
        @test all(d[1:2].data.data .== d.data.data)
        @test all(d[1:2].bags .== [1:4,0:-1])
        @test all(d[2].data.data .== d.data.data[:,0:-1])
        @test all(d[2].bags .== [0:-1])
    end

    @testset "testing WeightedBagNode indexing" begin
        @test all(wa[1].data.data .== wa.data.data)
        @test all(wa[1].bags .== [1:4])
        @test all(wb[1:2].data.data .== wb.data.data)
        @test all(wb[1:2].bags .== [1:2,3:4])
        @test all(wb[2].data.data .== wb.data.data[:,3:4])
        @test all(wb[2].bags .== [1:2])
        @test all(wb[1].data.data .== wb.data.data[:,1:2])
        @test all(wb[1].bags .== [1:2])
        @test all(wc[1].data.data .== wc.data.data[:,1:1])
        @test all(wc[1].bags .== [1:1])
        @test all(wc[[1,2]].data.data .== wc.data.data[:,1:2])
        @test all(wc[[1,2]].bags .== [1:1,2:2])
        @test all(wc[[2,1]].data.data .== wc.data.data[:,[2,1]])
        @test all(wc[[2,1]].bags .== [1:1,2:2])
        @test all(wd[[2,1]].data.data .== wd.data.data)
        @test all(wd[[2,1]].bags .== [0:-1,1:4])
        @test all(wd[1:2].data.data .== wd.data.data)
        @test all(wd[1:2].bags .== [1:4,0:-1])
        @test all(wd[2].data.data .== wd.data.data[:,0:-1])
        @test all(wd[2].bags .== [0:-1])
    end

    @testset "testing nested ragged array" begin
        x = BagNode(ArrayNode(rand(3,10)),[1:2,3:3,0:-1,4:5,6:6,7:10])
        y = BagNode(x,[1:2,3:3,4:5,6:6])
        @test all(y[1].data.data.data .== x.data.data[:,1:3])
        @test all(y[1].data.bags .== [1:2,3:3])
        @test all(y[1:2].data.data.data .== x.data.data[:,1:3])
        @test all(y[1:2].data.bags .== [1:2,3:3,0:-1])
        @test all(y[2:3].data.data.data .== x.data.data[:,4:6])
        @test all(y[2:3].data.bags .== [0:-1,1:2,3:3])
    end


    @testset "testing TreeNode" begin
        x = TreeNode((ArrayNode(rand(3,2)),ArrayNode(rand(3,2)),ArrayNode(randn(3,2))))
        y = TreeNode((ArrayNode(rand(3,2)),ArrayNode(rand(3,2)),ArrayNode(randn(3,2))))
        @test all(cat(x,y).data[1].data .== hcat(x.data[1].data,y.data[1].data))
        @test all(cat(x,y).data[2].data .== hcat(x.data[2].data,y.data[2].data))
        @test all(cat(x,y).data[3].data .== hcat(x.data[3].data,y.data[3].data))
    end

    @testset "testing sparsify" begin
        @test typeof(sparsify(zeros(10,10),0.05)) <: SparseMatrixCSC
        @test typeof(sparsify(randn(10,10),0.05)) <: Matrix
        @test typeof(sparsify(randn(10),0.05)) <: Vector
    end

    @testset "testing sparsify and mapdata" begin
        x = TreeNode((TreeNode((ArrayNode(randn(5,5)), ArrayNode(zeros(5,5)))), ArrayNode(zeros(5,5))))
        xs = mapdata(i -> sparsify(i,0.05), x)
        @test typeof(xs.data[2].data) <: SparseMatrixCSC
        @test typeof(xs.data[1].data[2].data) <: SparseMatrixCSC
        @test all(xs.data[1].data[1].data .== x.data[1].data[1].data)
    end

end
