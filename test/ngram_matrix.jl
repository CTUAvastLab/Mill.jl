@testset "NGramIterator and friends" begin
    for s in [randstring(100) for _ in 1:10]
        it = NGramIterator(s)
        @test length(it) == 100 + 3 - 1
        c = collect(it)
        @test c == ngrams(s)
        @test c == NGramIterator(codeunits(s)) |> collect
    end
end

@testset "NGramMatrix construction types and basics" begin
    M1 = NGramMatrix(["hello", "world"])
    M2 = NGramMatrix(["a part", "is", missing])
    M3 = NGramMatrix([missing, missing])
    @test M1 isa NGramMatrix{String}
    @test M1 isa AbstractMatrix{Int64}
    @test M2 isa NGramMatrix{Union{String, Missing}}
    @test M2 isa AbstractMatrix{Union{Int64, Missing}}
    @test M3 isa NGramMatrix{Missing}
    @test M3 isa AbstractMatrix{Missing}

    @test size(M1) == (2053, 2)
    @test size(M2) == (2053, 3)
    @test size(M3) == (2053, 2)

    M1 = NGramMatrix([codeunits("hello"), codeunits("world")])
    M2 = NGramMatrix([codeunits("a part"), codeunits("is"), missing])
    M3 = NGramMatrix([missing, missing])
    @test M1 isa NGramMatrix{<:CodeUnits}
    @test M1 isa AbstractMatrix{Int64}
    @test M2 isa NGramMatrix{<:Union{CodeUnits, Missing}}
    @test M2 isa AbstractMatrix{Union{Int64, Missing}}
    @test M3 isa NGramMatrix{Missing}
    @test M3 isa AbstractMatrix{Missing}

    @test size(M1) == (2053, 2)
    @test size(M2) == (2053, 3)
    @test size(M3) == (2053, 2)

    M1 = NGramMatrix([[1,2,3], [4,5,6]])
    M2 = NGramMatrix([Int[], [0], missing])
    M3 = NGramMatrix([missing, missing])
    @test M1 isa NGramMatrix{Vector{Int64}}
    @test M1 isa AbstractMatrix{Int64}
    @test M2 isa NGramMatrix{Union{Vector{Int64}, Missing}}
    @test M2 isa AbstractMatrix{Union{Int64, Missing}}
    @test M3 isa NGramMatrix{Missing}
    @test M3 isa AbstractMatrix{Missing}

    @test size(M1) == (2053, 2)
    @test size(M2) == (2053, 3)
    @test size(M3) == (2053, 2)
end

@testset "Indexing" begin
    S1 = ["hello", "world"]
    S2 = ["a part", "is", missing]
    S3 = [missing, missing]
    M1 = NGramMatrix(S1)
    M2 = NGramMatrix(S2)
    M3 = NGramMatrix(S3)

    for i in eachindex(S1)
        @test M1[:, i] == NGramMatrix(S1[i])
    end
    for i in eachindex(S2)
        @test isequal(M2[:, i], NGramMatrix(S2[i]))
    end
    for i in eachindex(S3)
        @test isequal(M3[:, i], NGramMatrix(S3[i]))
    end

    @test M1[:, [1,2]] == NGramMatrix(S1[[1,2]])
    @test isequal(M2[:, [3,2]], NGramMatrix(S2[[3,2]]))
    @test isequal(M3[:, [1]], NGramMatrix(S3[[1]]))

    @test M1[:, :] == M1
    @test isequal(M2[:, :], M2)
    @test isequal(M3[:, :], M3)
    @test M1[:, :] !== M1
    @test M2[:, :] !== M2
    @test M3[:, :] !== M3
    @test hash(M1[:, :]) == hash(M1)
    @test hash(M2[:, :]) == hash(M2)
    @test hash(M3[:, :]) == hash(M3)

    @test_throws MethodError M1[1, 2]
    @test_throws MethodError M1[1, :]
    @test_throws MethodError M1[[1,2], 2]
    @test_throws MethodError M1[[2,1], :]

    @test_throws BoundsError M1[:, 0]
    @test_throws BoundsError M1[:, 3]
    @test_throws BoundsError M1[:, [1,0]]
    @test_throws BoundsError M1[:, [1,2,3]]
end

@testset "hcat" begin
    S1 = ["hello", "world"]
    S2 = ["a part", "is", missing]
    S3 = [missing, missing]
    M1 = NGramMatrix(S1)
    M2 = NGramMatrix(S2)
    M3 = NGramMatrix(S3)

    @test isequal(hcat(M1, M2), NGramMatrix(vcat(S1, S2)))
    @test isequal(hcat(M1, M3), NGramMatrix(vcat(S1, S3)))
    @test isequal(hcat(M2, M3), NGramMatrix(vcat(S2, S3)))
    @test isequal(hcat(M1, M2, M3), NGramMatrix(vcat(S1, S2, S3)))

    @test_throws DimensionMismatch hcat(NGramMatrix(S1, 1), NGramMatrix(S1, 2))
    @test_throws DimensionMismatch hcat(NGramMatrix(S1, 2, 256), NGramMatrix(S1, 2, 128))
    @test_throws DimensionMismatch hcat(NGramMatrix(S1, 2, 256, 7), NGramMatrix(S1, 2, 256, 2053))
end

@testset "ngram computation" begin
    x = [1,3,5,2,6,8,3]
    b = 8 + 1

    slicer = (x,n) -> map(i -> vcat(fill(Mill.string_start_code(), max(0, n-i)),
                                    x[max(i-n+1,1):min(i,length(x))],
                                    fill(Mill.string_end_code(), max(0, i-length(x)))),
                                    1:length(x)+n-1)
    indexes = (x,b) -> mapreduce(i -> i[2] * b^(i[1]-1), +, enumerate(reverse(x)))
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
            @test all(countngrams(s,3,256,10) .== idx2vec(ngrams(s,3,256),10))
        end

        s = split("Lorem ipsum dolor sit amet, consectetur adipiscing elit")
        @test all(countngrams(s,3,256,10) .== hcat(map(ss -> idx2vec(ngrams(ss,3,256), 10),s)...))
    end
end

@testset "string2ngrams" begin
    @test size(string2ngrams(["", "a"], 3, 256, 2053)) == (2053,2)
end

@testset "NGramMatrix to SparseMatrix" begin
    for (n, m) in product([2,3,5], [10, 100, 1000])
        b = 256
        s = ["hello", "world", "!!!"]
        si = map(codeunits, s)
        sc = map(i -> Int.(i), si)
        B = NGramMatrix(s, n, b, m)
        Bi = NGramMatrix(si, n, b, m)
        Bc = NGramMatrix(sc, n, b, m)
        @test SparseMatrixCSC(B) == SparseMatrixCSC(Bi) == SparseMatrixCSC(Bc)
        @test SparseMatrixCSC(string2ngrams(s, n, b, m)) == SparseMatrixCSC(B)
        @test SparseMatrixCSC(string2ngrams(si, n, b, m)) == SparseMatrixCSC(Bi)
        @test SparseMatrixCSC(string2ngrams(sc, n, b, m)) == SparseMatrixCSC(Bc)
    end
end

@testset "NGramMatrix multiplication" begin
    for (n, m) in product([2,3,5], [10, 100, 1000])
        b = 256
        A = randn(10, m)

        s = ["hello", "world", "!!!"]
        si = map(codeunits, s)
        sc = map(i -> Int.(i), si)
        B = NGramMatrix(s, n, b, m)
        Bi = NGramMatrix(si, n, b, m)
        Bc = NGramMatrix(sc, n, b, m)

        @test all(A * B ≈ A * string2ngrams(s, n, b, m))
        @test all(A * Bi ≈ A * B)
        @test all(A * Bc ≈ A * B)

        @test all(A * B ≈ A * SparseMatrixCSC(B))
        @test all(A * B ≈ A * SparseMatrixCSC(Bi))
        @test all(A * B ≈ A * SparseMatrixCSC(Bc))

        @inferred A * B
        @inferred A * Bi
        @inferred A * Bc

        @test eltype(A * B) == eltype(A)
        @test eltype(A * Bi) == eltype(A)
        @test eltype(A * Bc) == eltype(A)
    end
end

@testset "NGramMatrix multiplication gradtest" begin
    for (n, m) in product([2,3,5], [10, 100, 1000])
        b = 256
        A = randn(10, m)

        s = ["hello", "world", "!!!"]
        si = map(codeunits, s)
        sc = map(i -> Int.(i), si)
        Ns = string2ngrams(s, n, b, m)
        Nsi = string2ngrams(si, n, b, m)
        Nsc = string2ngrams(sc, n, b, m)
        B = NGramMatrix(s, n, b, m)
        Bi = NGramMatrix(si, n, b, m)
        Bc = NGramMatrix(sc, n, b, m)
        Ss = SparseMatrixCSC(B)
        Ssi = SparseMatrixCSC(Bi)
        Ssc = SparseMatrixCSC(Bc)

        dA, dB = gradient(sum ∘ *, A, B)
        @test dA ≈ gradient(A -> sum(A * B), A) |> only
        @test dA ≈ gradient(A -> sum(A * Ns), A) |> only
        @test dA ≈ gradient(A -> sum(A * Ss), A) |> only
        @test gradtest(A -> sum(A * B), A)

        @test dB === gradient(B -> sum(A * B), B) |> only
        @test isnothing(dB)

        dA, dBi = gradient(sum ∘ *, A, Bi)
        @test dA ≈ gradient(A -> sum(A * Bi), A) |> only
        @test dA ≈ gradient(A -> sum(A * Ns), A) |> only
        @test dA ≈ gradient(A -> sum(A * Ss), A) |> only
        @test gradtest(A -> sum(A * Bi), A)

        @test dBi === gradient(Bi -> sum(A * Bi), Bi) |> only
        @test isnothing(dBi)

        dA, dBc = gradient(sum ∘ *, A, Bc)
        @test dA ≈ gradient(A -> sum(A * Bc), A) |> only
        @test dA ≈ gradient(A -> sum(A * Ns), A) |> only
        @test dA ≈ gradient(A -> sum(A * Ss), A) |> only
        @test gradtest(A -> sum(A * Bc), A)

        @test dBc === gradient(Bc -> sum(A * Bc), Bc) |> only
        @test isnothing(dBc)
    end
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

@testset "equals with missings" begin
    M1 = NGramMatrix(["hello", "world"])
    M2 = NGramMatrix(["a part", "is", missing])
    M3 = NGramMatrix([missing, missing])
    @test M1 == M1
    @test isequal(M1, M1)
    @test M2 != M2
    @test isequal(M2, M2)
    @test M3 != M3
    @test isequal(M3, M3)
    @test M1 != M2
    @test !isequal(M1, M2)
    @test M1 != M3
    @test !isequal(M1, M3)
    @test M2 != M3
    @test !isequal(M2, M3)
end

begin
    println("Benchmarking multiplication")
    # begin block body
    A = randn(80,2053);
    s = [randstring(10) for i in 1:1000];
    B = NGramMatrix(s, 3, 256, 2053)
    C = sparse(string2ngrams(s, 3, 256, size(A, 2)));
    println("A * B::NGramMatrix (This should be the fastest)");
    @btime A*B;                                                 # 526.456 μs (2002 allocations: 671.95 KiB)
    println("A * string2ngrams(s, 3, 256, size(A, 2))")
    @btime A*string2ngrams(s, 3, 256, size(A, 2));                   # 154.646 ms (3013 allocations: 16.38 MiB)
    println("A * sparse(string2ngrams(s, 3, 256, size(A, 2)))")
    @btime A*sparse(string2ngrams(s, 3, 256, size(A, 2)));           # 7.525 ms (3013 allocations: 16.57 MiB)
    print("A * C where C = sparse(string2ngrams(s, 3, 256, size(A, 2)));");
    @btime A*C;                                                 # 1.527 ms (2 allocations: 625.08 KiB)
end
