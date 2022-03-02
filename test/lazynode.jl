@testset "LazyNode" begin
    ss = ["Hello world.", "Make peace.", "Make food.", "Eat penam."]

    @test LazyNode{:Sentence}(ss).data == ss
    @test LazyNode(:Sentence, ss).data == ss
    @test LazyNode{:Sentence}(ss).data == ss
    @test nobs(LazyNode{:Sentence}(ss)) == 4
    @test (LazyNode{:Sentence}(ss))[[1,3]].data == ss[[1,3]]
    @test (LazyNode{:Sentence}(ss))[2].data == [ss[2]]
    @test catobs((LazyNode{:Sentence}(ss)[i] for i in [1:2,3:4])...).data == ss
    @test catobs((LazyNode{:Sentence}(ss)[i] for i in [1,2,3,4])...).data == ss
    @test reduce(catobs, [LazyNode{:Sentence}(ss)[i] for i in [1:2,3:4]]).data == ss
    @test reduce(catobs, [LazyNode{:Sentence}(ss)[i] for i in [1,2,3,4]]).data == ss

    ds = LazyNode{:Sentence}(ss)
    m = Mill.reflectinmodel(ds)
    @test m(ds) ≈ m.m(Mill.unpack2mill(ds))
end
