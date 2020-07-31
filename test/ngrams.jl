using Test
using Mill, SparseArrays, Random, Flux
using Mill: NGramIterator, ngrams, string2ngrams, countngrams, mul, multrans, NGramMatrix, catobs
import BenchmarkTools: @btime

@testset "ngrams" begin
    x = [1,3,5,2,6,8,3]
    b = 8 + 1

    slicer = (x,n) -> map(i -> x[(max(i-n+1,1):min(i,length(x)))],1:length(x)+n-1)
    indexes = (x,b) -> mapreduce(i -> i[2]*b^(i[1]-1), +, enumerate(reverse(x)))
    function idx2vec(i,n)
        o = zeros(Int,n)
        for v in i
            o[mod(v,n)+1] +=1
        end
        o
    end

    @testset "testing ngrams on vector of Ints" begin
        @test all(ngrams(x,3,b) .== map(x -> indexes(x,b),slicer(x,3)))
        @test all(ngrams(x,2,b) .== map(x -> indexes(x,b),slicer(x,2)))
        @test all(ngrams(x,1,b) .== map(x -> indexes(x,b),slicer(x,1)))
    end

    @testset "testing frequency of ngrams on vector of Ints and on Strings" begin
        @test all(countngrams(x,3,b,10) .== idx2vec(map(x -> indexes(x,b), slicer(x,3)), 10))
        for s in split("Lorem ipsum dolor sit amet, consectetur adipiscing elit")
            @test all(countngrams(s,3,256,10) .== idx2vec(ngrams(s,3,256), 10))
        end

        s = split("Lorem ipsum dolor sit amet, consectetur adipiscing elit")
        @test all(countngrams(s,3,256,10) .== hcat(map(ss -> idx2vec(ngrams(ss,3,256), 10),s)...))
    end


    @testset "string2ngrams" begin
        @test size(string2ngrams(["","a"],3,2053)) == (2053,2)
    end

    @testset "NGramMatrix" begin
        @test all(collect(NGramIterator(codeunits("hello"), 3, 257)) .== ngrams("hello", 3, 257))

        A = randn(4, 10)
        n = size(A, 2)
        s = ["hello", "world", "!!!"]
        si = map(i -> Int.(codeunits(i)), s)
        B = NGramMatrix(s, 3, 256, n)
        Bi = NGramMatrix(si, 3, 256, n)
        @test all(A * B ≈ A*string2ngrams(s, 3, n))
        @test all(A * Bi ≈ A * B)
        A = randn(5,3)
        @test all(multrans(A , B) ≈ A*transpose(string2ngrams(s, 3, n)))
        @test all(multrans(A , B) ≈ multrans(A , B))
    end

    @testset "NGramMatrix to SparseMatrix" begin
        A = randn(4, 10)
        n = size(A, 2)
        s = ["hello", "world", "!!!"]
        si = map(i -> Int.(codeunits(i)), s)
        B = NGramMatrix(s, 3, 256, n)
        Bi = NGramMatrix(si, 3, 256, n)
        @test all(A * B ≈  A * SparseMatrixCSC(B))
        @test all(A * B ≈  A * SparseMatrixCSC(Bi))
    end

    @testset "integration with MILL & Flux" begin
        s = ["hello", "world", "!!!"]
        si = map(i -> Int.(codeunits(i)), s)
        for (a, s) in [(NGramMatrix(s, 3, 256, 2057), s), (NGramMatrix(si, 3, 256, 2057), si)]
            @test all(reduce(catobs, [a, a]).s .== vcat(s,s))
            @test all(hcat(a,a).s .== vcat(s,s))

            W = randn(40, 2057)
            @test gradcheck(x -> sum(x * a), W)

            a = ArrayNode(a, nothing)
            @test all(reduce(catobs, [a, a]).data.s .== vcat(s,s))
            a = BagNode(a, [1:3], nothing)
            @test all(reduce(catobs, [a, a]).data.data.s .== vcat(s,s))
        end
    end
end

begin
    println("Benchmarking multiplication")
    # begin block body
    A = randn(80,2053);
    s = [randstring(10) for i in 1:1000];
    B = NGramMatrix(s, 3, 256, 2053)
    C = sparse(string2ngrams(s, 3, size(A, 2)));
    println("A * B::NGramMatrix (This should be the fastest)");
    @btime A*B;                                                 # 526.456 μs (2002 allocations: 671.95 KiB)
    println("A * string2ngrams(s, 3, size(A, 2))")
    @btime A*string2ngrams(s, 3, size(A, 2));                   # 154.646 ms (3013 allocations: 16.38 MiB)
    println("A * sparse(string2ngrams(s, 3, size(A, 2)))")
    @btime A*sparse(string2ngrams(s, 3, size(A, 2)));           # 7.525 ms (3013 allocations: 16.57 MiB)
    print("A * C where C = sparse(string2ngrams(s, 3, size(A, 2)));");
    @btime A*C;                                                 # 1.527 ms (2 allocations: 625.08 KiB)
end
