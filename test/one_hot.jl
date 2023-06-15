@testset "onehot hcat optimizations" begin
    X1 = onehotbatch(collect(1:5), 1:10)
    @test @which(hcat(X1, X1)).module ≡ OneHotArrays
    @test @which(reduce(hcat, [X1, X1])).module ≡ Mill

    @test hcat(X1, X1) == onehotbatch(vcat(1:5, 1:5), 1:10)
    @test reduce(hcat, [X1, X1]) == onehotbatch(vcat(1:5, 1:5), 1:10)
    @test reduce(catobs, [X1, X1]) == onehotbatch(vcat(1:5, 1:5), 1:10)

    X2 = OneHotArrays.onehotbatch(collect(1:6), 1:12)
    @test_throws DimensionMismatch hcat(X1, X2)
end
