@testset "NGramIterator and friends" begin
    for s in [randustring(100) for _ in 1:10]
        it = NGramIterator(s)
        @test length(it) == length(codeunits(s)) + 3 - 1
        c = collect(it)
        @test c == ngrams(s)
        @test c == NGramIterator(codeunits(s)) |> collect
    end
end

@testset "NGramMatrix construction types and basics" begin
    M1 = NGramMatrix(["heλlo", "world"])
    M2 = NGramMatrix(["α part", "is", missing] |> PooledArray)
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

    M1 = NGramMatrix([codeunits("heλlo"), codeunits("world")])
    M2 = NGramMatrix([codeunits("α part"), codeunits("is"), missing])
    M3 = NGramMatrix([missing, missing] |> PooledArray)
    @test M1 isa NGramMatrix{<:CodeUnits}
    @test M1 isa AbstractMatrix{Int64}
    @test M2 isa NGramMatrix{<:Union{CodeUnits, Missing}}
    @test M2 isa AbstractMatrix{Union{Int64, Missing}}
    @test M3 isa NGramMatrix{Missing}
    @test M3 isa AbstractMatrix{Missing}

    @test size(M1) == (2053, 2)
    @test size(M2) == (2053, 3)
    @test size(M3) == (2053, 2)

    M1 = NGramMatrix([[1,2,3], [4,5,6]] |> PooledArray)
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
    S1 = ["heλlo", "world"]
    S2 = ["α part", "is", missing] |> PooledArray
    S3 = [missing, missing]
    M1 = NGramMatrix(S1)
    M2 = NGramMatrix(S2)
    M3 = NGramMatrix(S3)

    for i in eachindex(S1)
        @test M1[:, i] == NGramMatrix(S1[i])
    end
    for i in eachindex(S2)
        @test isequal(M2[:, i], NGramMatrix([S2[i]] |> PooledArray))
    end
    for i in eachindex(S3)
        @test isequal(M3[:, i], NGramMatrix(S3[i]))
    end

    @test M1[:, [1,2]] == M1[:, 1:2] == NGramMatrix(S1[[1,2]])
    @test isequal(M2[:, [3,2]], NGramMatrix(S2[[3,2]]))
    @test areequal(M3[:, [1]], M3[:, 1:1], NGramMatrix(S3[1]))
    @test areequal(M3[:, Int[]], M3[:, 1:0], NGramMatrix(String[]))

    @test M1[:, :] == M1
    @test isequal(M2[:, :], M2)
    @test isequal(M3[:, :], M3)
    @test M1[:, :] !== M1
    @test M2[:, :] !== M2
    @test M3[:, :] !== M3
    @test hash(M1[:, :]) == hash(M1)
    @test hash(M2[:, :]) == hash(M2)
    @test hash(M3[:, :]) == hash(M3)

    @test_throws MethodError M1[1]
    @test_throws MethodError M1[[1, 2]]
    @test_throws MethodError M1[1:2]

    @test_throws MethodError M1[1, 2]
    @test_throws MethodError M1[1, :]
    @test_throws MethodError M1[[1,2], 2]
    @test_throws MethodError M1[[2,1], :]

    @test_throws BoundsError M1[:, 0]
    @test_throws BoundsError M1[:, 0:1]
    @test_throws BoundsError M1[:, 3]
    @test_throws BoundsError M1[:, 2:3]
    @test_throws BoundsError M1[:, [1,0]]
    @test_throws BoundsError M1[:, [1,2,3]]
end

@testset "hcat" begin
    S1 = ["heλlo", "world"]
    S2 = ["α part", "is", missing]
    S3 = [missing, missing]
    M1 = NGramMatrix(S1)
    M2 = NGramMatrix(S2)
    M3 = NGramMatrix(S3)

    @test areequal(hcat(M1, M2), reduce(hcat, [M1, M2]), NGramMatrix(vcat(S1, S2)))
    @test areequal(hcat(M1, M3), reduce(hcat, [M1, M3]), NGramMatrix(vcat(S1, S3)))
    @test areequal(hcat(M2, M3), reduce(hcat, [M2, M3]), NGramMatrix(vcat(S2, S3)))
    @test areequal(hcat(M1, M2, M3), reduce(hcat, [M1, M2, M3]), NGramMatrix(vcat(S1, S2, S3)))

    @inferred hcat(M1, M1)
    @inferred reduce(hcat, [M1, M1])
    @inferred hcat(M2, M2)
    @inferred reduce(hcat, [M2, M2])
    @inferred hcat(M3, M3)
    @inferred reduce(hcat, [M3, M3])
    @inferred hcat(M1, M2)
    @inferred reduce(hcat, [M1, M2])
    @inferred hcat(M2, M3)
    @inferred reduce(hcat, [M2, M3])
    @inferred hcat(M3, M1)
    @inferred reduce(hcat, [M3, M1])
    @inferred hcat(M1, M2, M3)
    @inferred reduce(hcat, [M1, M2, M3])

    @test_throws DimensionMismatch hcat(NGramMatrix(S1, 1), NGramMatrix(S1, 2))
    @test_throws DimensionMismatch hcat(NGramMatrix(S1, 2, 256), NGramMatrix(S1, 2, 128))
    @test_throws DimensionMismatch hcat(NGramMatrix(S1, 2, 256, 7), NGramMatrix(S1, 2, 256, 2053))
    @test_throws DimensionMismatch reduce(hcat, [NGramMatrix(S1, 1), NGramMatrix(S1, 2)])
    @test_throws DimensionMismatch reduce(hcat, [NGramMatrix(S1, 2, 256), NGramMatrix(S1, 2, 128)])
    @test_throws DimensionMismatch reduce(hcat, [NGramMatrix(S1, 2, 256, 7), NGramMatrix(S1, 2, 256, 2053)])
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

    @testset "ngrams on vector of Ints" begin
        @test ngrams(x,3,b) == map(x -> indexes(x,b),slicer(x,3))
        @test ngrams(x,2,b) == map(x -> indexes(x,b),slicer(x,2))
        @test ngrams(x,1,b) == map(x -> indexes(x,b),slicer(x,1))
    end

    @testset "frequency of ngrams on vector of Ints and on Strings" begin
        @test countngrams(x,3,b,10) == idx2vec(map(x -> indexes(x,b), slicer(x,3)), 10)
        for s in split("Lorem ipsum dolor sit amet, consectetur adipiscing elit")
            @test countngrams(s,3,256,10) == idx2vec(ngrams(s,3,256),10)
        end

        s = split("Lorem ipsum dolor sit amet, consectetur adipiscing elit")
        @test countngrams(s,3,256,10) == hcat(map(ss -> idx2vec(ngrams(ss,3,256), 10),s)...)
    end
end

@testset "NGramMatrix to SparseMatrix" begin
    for (n, m) in product([2,3,5], [10, 100, 1000])
        b = 256
        S = ["heλlo", "world", "!!!"]
        Si = map(codeunits, S)
        Sc = map(i -> Int.(i), Si)
        B = NGramMatrix(S, n, b, m)
        Bi = NGramMatrix(Si, n, b, m)
        Bc = NGramMatrix(Sc, n, b, m)
        @test SparseMatrixCSC(B) == SparseMatrixCSC(Bi) == SparseMatrixCSC(Bc)
        @test SparseMatrixCSC(countngrams(S, n, b, m)) == SparseMatrixCSC(B)
        @test SparseMatrixCSC(countngrams(Si, n, b, m)) == SparseMatrixCSC(Bi)
        @test SparseMatrixCSC(countngrams(Sc, n, b, m)) == SparseMatrixCSC(Bc)
    end
end

@testset "NGramMatrix multiplication" begin
    for (n, m) in product([2,3,5], [10, 100, 1000])
        b = 256
        A = randn(10, m)

        S = [randustring(100) for _ in 1:10]
        Si = map(codeunits, S)
        Sc = map(i -> Int.(i), Si)
        B = NGramMatrix(S, n, b, m)
        Bi = NGramMatrix(Si, n, b, m)
        Bc = NGramMatrix(Sc, n, b, m)

        @test all(A * B ≈ A * countngrams(S, n, b, m))
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
    for (n, m) in product([2,3,5], [10, 20])
        b = 256
        A = randn(10, m)

        s = [randustring(100) for _ in 1:10] |> PooledArray
        si = map(codeunits, s)
        sc = map(i -> Int.(i), si)
        Ns = countngrams(s, n, b, m)
        Nsi = countngrams(si, n, b, m)
        Nsc = countngrams(sc, n, b, m)
        B = NGramMatrix(s, n, b, m)
        Bi = NGramMatrix(si, n, b, m)
        Bc = NGramMatrix(sc, n, b, m)
        Ss = SparseMatrixCSC(B)
        Ssi = SparseMatrixCSC(Bi)
        Ssc = SparseMatrixCSC(Bc)

        _, f = gradf(*, A, B)
        dNs = gradient(f, A, Ns)[1]
        dSs = gradient(f, A, Ss)[1]

        dA, dB = gradient(f, A, B)
        @test dA ≈ dNs
        @test dA ≈ dSs
        @test isnothing(dB)
        @test @gradtest A -> A * B

        dA, dBi = gradient(f, A, Bi)
        @test dA ≈ dNs
        @test dA ≈ dSs
        @test isnothing(dBi)
        @test @gradtest A -> A * Bi

        dA, dBc = gradient(f, A, Bc)
        @test dA ≈ dNs
        @test dA ≈ dSs
        @test isnothing(dBi)
        @test @gradtest A -> A * Bc
    end
end

@testset "integration with Mill & Flux" begin
    S = ["heλlo", "world", "!!!"]
    Si = map(i -> Int.(codeunits(i)), S)
    for (A, S) in [(NGramMatrix(S, 3, 256, 2057), S), (NGramMatrix(Si, 3, 256, 2057), Si)]
        @test reduce(catobs, [A, A]).S == vcat(S, S)
        @test hcat(A, A).S == vcat(S, S)

        W = randn(40, 2057)
        @test @gradtest W -> W * A

        n = ArrayNode(A, nothing)
        @test reduce(catobs, [n, n]).data.S == vcat(S, S)
        n = BagNode(n, [1:3], nothing)
        @test reduce(catobs, [n, n]).data.data.S == vcat(S, S)
    end
end

@testset "equals with missings" begin
    M1 = NGramMatrix(["heλlo", "world"])
    M2 = NGramMatrix(["α part", "is", missing])
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
    A = randn(80,2053);
    S = [randustring(10) for i in 1:1000];
    B = NGramMatrix(S, 3, 256, 2053)
    C = sparse(countngrams(S, 3, 256, size(A, 2)));
    println("A * B::NGramMatrix (This should be the fastest)");
    @btime A*B;                                                 # 501.238 μs (2 allocations: 625.08 KiB)
    println("A * countngrams(S, 3, 256, size(A, 2))")
    @btime A * countngrams(S, 3, 256, size(A, 2));              # 138.392 ms (11 allocations: 16.27 MiB)
    println("A * sparse(countngrams(S, 3, 256, size(A, 2)))")
    @btime A*sparse(countngrams(S, 3, 256, size(A, 2)));        # 5.241 ms (11 allocations: 16.46 MiB)
    print("A * C where C = sparse(countngrams(S, 3, 256, size(A, 2)));");
    @btime A*C;                                                 # 350.919 μs (2 allocations: 625.08 KiB)
end
