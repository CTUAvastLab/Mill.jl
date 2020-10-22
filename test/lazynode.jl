using Mill, Flux, Test
using Mill: LazyNode

@testset "LazyNode" begin
    ss = ["Hello world.", "Make peace.", "Make food.", "Eat penam."]

    function Mill.unpack2mill(ds::LazyNode{:Sentence})
        s = ds.data 
        ss = map(x -> split(x, " "),s)
        x = NGramMatrix(reduce(vcat, ss), 3, 256, 2053)
        BagNode(ArrayNode(x), Mill.length2bags(length.(ss)))
    end

    @test LazyNode{:Sentence,Vector{String}}(ss).data == ss
    @test LazyNode(:Sentence, ss).data == ss
    @test LazyNode{:Sentence}(ss).data == ss
    @test nobs(LazyNode{:Sentence}(ss)) == 4
    @test (LazyNode{:Sentence}(ss))[[1,3]].data == ss[[1,3]]
    @test (LazyNode{:Sentence}(ss))[2].data == [ss[2]]
    @test reduce(catobs, [LazyNode{:Sentence}(ss)[i] for i in [1:2,3:4]]).data == ss
    @test reduce(catobs, [LazyNode{:Sentence}(ss)[i] for i in [1,2,3,4]]).data == ss

    ds = LazyNode{:Sentence}(ss)
    m = Mill.reflectinmodel(ds, d -> Dense(d,2), s -> SegmentedMeanMax(s))
    @test m(ds).data ≈ m.m(Mill.unpack2mill(ds)).data
end
