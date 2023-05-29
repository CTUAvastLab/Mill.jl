@testset "cat" begin
    Ws = [randn(3,rand(1:10)) for _ in 1:4]
    ψs = [randn(size(Ws[i], 2)) for i in 1:4]
    As = PreImputingMatrix.(Ws, ψs)
    for is in powerset(1:4), is in permutations(is)
        length(is) > 0 || continue
        @test hcat(As[is]...) == PreImputingMatrix(hcat(Ws[is]...), vcat(ψs[is]...))
        @inferred hcat(As[is]...)
        @test_throws ArgumentError vcat(As[is]...)
    end

    Ws = [randn(rand(1:10), 3) for _ in 1:4]
    ψs = [randn(size(Ws[i], 1)) for i in 1:4]
    As = PostImputingMatrix.(Ws, ψs)
    for is in powerset(1:4), is in permutations(is)
        length(is) > 0 || continue
        @test vcat(As[is]...) == PostImputingMatrix(vcat(Ws[is]...), vcat(ψs[is]...))
        @inferred vcat(As[is]...)
        @test_throws ArgumentError hcat(As[is]...)
    end
end

@testset "Wrong dimensions" begin
    A = PreImputingMatrix(rand(2, 3), rand(3))
    @test_throws DimensionMismatch A * []
    @test_throws DimensionMismatch A * [[]]
    @test_throws DimensionMismatch A * [1, 2]
    @test_throws DimensionMismatch A * [missing, missing]
    @test_throws DimensionMismatch A * [1, 2, 3, 4]
    @test_throws DimensionMismatch A * [1, 2, 3, missing]

    A = PostImputingMatrix(rand(2, 3), rand(2))
    @test_throws DimensionMismatch A * MaybeHotVector(2, 2)
    @test_throws DimensionMismatch A * MaybeHotVector(missing, 4)
    @test_throws DimensionMismatch A * MaybeHotMatrix(Int[], 1)
    @test_throws DimensionMismatch A * MaybeHotMatrix([1, 2], 5)
    @test_throws DimensionMismatch A * MaybeHotMatrix([missing, 5, 8], 10)
    @test_throws DimensionMismatch A * MaybeHotMatrix([missing, missing], 2)
end

@testset "correct pre imputing behavior for standard vector (maybe missing)" begin
    function _test_imput(W, ob::Vector, b::Vector)
        A = PreImputingMatrix(W, ob)
        @test A * b == W * ob
        @test eltype(A * b) ≡ promote_type(eltype(W), eltype(ob))
        @inferred A * b
    end

    function _test_imput(W, ob::Vector, t=10, p=0.2)
        for _ in 1:t
            b = [rand() < p ? missing : x for x in ob]
            _test_imput(W, ob, b)
        end
    end

    _test_imput(randn(3, 3), randn(3))
    _test_imput(reshape(1:9 |> collect, (3, 3)), rand(1:3, 3))

    _test_imput(randn(3, 3), randn(3), fill(missing, 3))
    _test_imput(reshape(1:9 |> collect, (3, 3)), rand(1:3, 3), fill(missing, 3))
end

@testset "correct type promotion for pre imputing of standard vector" begin
    A = PreImputingMatrix(rand(1:3, 2, 3), rand(1:3, 3))
    @test eltype(A * rand(1:3, 3)) ≡ Int
    @test eltype(A * [1, missing, 3]) ≡ Int
    @test eltype(A * rand(3)) ≡ Float64
    @test eltype(A * [1.0, missing, 3]) ≡ Float64
    @test eltype(A * rand(Float32, 3)) ≡ Float32
    @test eltype(A * Vector{Maybe{Float32}}(rand(3))) ≡ Float32
    @test eltype(A * fill(missing, 3)) ≡ Int

    A = PreImputingMatrix(rand(2, 3), rand(3))
    @test eltype(A * rand(1:3, 3)) ≡ Float64
    @test eltype(A * [1, missing, 3]) ≡ Float64
    @test eltype(A * rand(3)) ≡ Float64
    @test eltype(A * [1.0, missing, 3]) ≡ Float64
    @test eltype(A * rand(Float32, 3)) ≡ Float64
    @test eltype(A * Vector{Maybe{Float32}}(rand(3))) ≡ Float64
    @test eltype(A * fill(missing, 3)) ≡ Float64

    A = PreImputingMatrix(rand(Float32, 2, 3), rand(Float32, 3))
    @test eltype(A * rand(1:3, 3)) ≡ Float32
    @test eltype(A * [1, missing, 3]) ≡ Float32
    @test eltype(A * rand(3)) ≡ Float64
    @test eltype(A * [1.0, missing, 3]) ≡ Float64
    @test eltype(A * rand(Float32, 3)) ≡ Float32
    @test eltype(A * Vector{Maybe{Float32}}(rand(3))) ≡ Float32
    @test eltype(A * fill(missing, 3)) ≡ Float32
end

@testset "correct pre imputing behavior for standard matrix (maybe missing)" begin
    W = [1 2; 3 4]
    ψ = [2, 1]
    A = PreImputingMatrix(W, ψ)
    B1 = [4 3; 2 1]
    B2 = [missing missing; missing missing]
    B3 = [missing 3; 1 missing]
    B4 = [3 missing; missing 1]

    @test A * B1 == W * B1
    @test A * B2 == [W * ψ W * ψ]
    @test A * B3 == W * [2 3; 1 1]
    @test A * B4 == W * [3 2; 1 1]

    @inferred A * B1
    @inferred A * B2
    @inferred A * B3
    @inferred A * B4

    @test eltype(A * B1) ≡ eltype(A * B2) ≡ eltype(A * B3) ≡ eltype(A * B4) ≡ Int64
end

@testset "correct type promotion for pre imputing of standard matrix" begin
    A = PreImputingMatrix(rand(1:3, 2, 3), rand(1:3, 3))
    @test eltype(A * rand(1:3, 3, 2)) ≡ Int
    @test eltype(A * [1 missing; 3 missing; 5 6]) ≡ Int
    @test eltype(A * rand(3, 2)) ≡ Float64
    @test eltype(A * [1.0 missing; 3 missing; 5 6]) ≡ Float64
    @test eltype(A * rand(Float32, 3, 2)) ≡ Float32
    @test eltype(A * Matrix{Maybe{Float32}}(rand(3, 2))) ≡ Float32
    @test eltype(A * fill(missing, 3, 2)) ≡ Int

    A = PreImputingMatrix(rand(2, 3), rand(3))
    @test eltype(A * rand(1:3, 3, 2)) ≡ Float64
    @test eltype(A * [1 missing; 3 missing; 5 6]) ≡ Float64
    @test eltype(A * rand(3, 2)) ≡ Float64
    @test eltype(A * [1.0 missing; 3 missing; 5 6]) ≡ Float64
    @test eltype(A * rand(Float32, 3, 2)) ≡ Float64
    @test eltype(A * Matrix{Maybe{Float32}}(rand(3, 2))) ≡ Float64
    @test eltype(A * fill(missing, 3, 2)) ≡ Float64

    A = PreImputingMatrix(rand(Float32, 2, 3), rand(Float32, 3))
    @test eltype(A * rand(1:3, 3, 2)) ≡ Float32
    @test eltype(A * [1 missing; 3 missing; 5 6]) ≡ Float32
    @test eltype(A * rand(3, 2)) ≡ Float64
    @test eltype(A * [1.0 missing; 3 missing; 5 6]) ≡ Float64
    @test eltype(A * rand(Float32, 3, 2)) ≡ Float32
    @test eltype(A * Matrix{Maybe{Float32}}(rand(3, 2))) ≡ Float32
    @test eltype(A * fill(missing, 3, 2)) ≡ Float32
end

@testset "correct post imputing behavior for full standard arrays" begin
    W = [1 2; 3 4]
    ψ = [2, 1]
    A = PostImputingMatrix(W, ψ)
    B = [4 3; 2 1]
    b = [3, 1]
    @test A * B == W * B
    @test A * b == W * b
    @inferred A * B
    @inferred A * b
    @test eltype(A * B) ≡ Int64
    @test eltype(A * b) ≡ Int64

    B = [missing 3; missing 1]
    b = [3, missing]
    @test isequal(A * B, W * B)
    @test isequal(A * b, W * b)
    @inferred A * B
    @inferred A * b
    @test eltype(A * B) ≡ Maybe{Int64}
    @test eltype(A * b) ≡ Maybe{Int64}

    W = randn(2,3)
    ψ = randn(2)
    A = PostImputingMatrix(W, ψ)

    B = randn(3, 5)
    b = randn(3)
    @test A * B ≈ W * B
    @test A * b ≈ W * b
    @inferred A * B
    @inferred A * b
    @test eltype(A * B) ≡ Float64
    @test eltype(A * b) ≡ Float64

    B = [missing 3.0; missing 1.0; 2.0 5.0]
    b = [3.0, missing, 1.0]
    @test isequal(A * B, W * B)
    @test isequal(A * b, W * b)
    @inferred A * B
    @inferred A * b
    @test eltype(A * B) ≡ Maybe{Float64}
    @test eltype(A * b) ≡ Maybe{Float64}

    B = fill(missing, 3, 5)
    b = fill(missing, 3)
    @test isequal(A * B, W * B)
    @test isequal(A * b, W * b)
    @inferred A * B
    @inferred A * b
    @test eltype(A * B) ≡ Missing
    @test eltype(A * b) ≡ Missing
end

@testset "correct post imputing behavior for maybe hot vector" begin
    for (m, n) in product(fill((2, 5, 10), 2)...)
        W = randn(m, n)
        ψ = randn(m)
        A = PostImputingMatrix(W, ψ)
        i = rand(1:n)
        b = MaybeHotVector(i, n)
        @test A * b == W * onehot(b)
        @inferred A * b
        @test eltype(A * b) ≡ eltype(W)
        b = MaybeHotVector(missing, n)
        @test A * b == ψ
        @inferred A * b
        @test eltype(A * b) ≡ eltype(ψ)
    end
end

@testset "correct post imputing behavior for maybe hot matrix" begin
    for (m, n, k) in product(fill((2, 5, 10), 3)...)
        W = randn(m, n)
        ψ = randn(m)
        A = PostImputingMatrix(W, ψ)
        i1 = rand(1:n, k)
        i2 = fill(missing, k)
        i3 = [isodd(i) ? missing : rand(1:n) for i in 1:k]
        B1 = MaybeHotMatrix(i1, n)
        B2 = MaybeHotMatrix(i2, n)
        B3 = MaybeHotMatrix(i3, n)
        @test A * B1 == W * onehotbatch(B1)
        @inferred A * B1
        @test eltype(A * B1) ≡ eltype(W)
        @test all(isequal(ψ), eachcol(A * B2))
        @inferred A * B2
        @test eltype(A * B2) ≡ eltype(ψ)
        C = A * B3
        @inferred A * B3
        @test eltype(A * B3) ≡ promote_type(eltype(W), eltype(ψ))
        @test all(isequal(ψ), eachcol(C[:, ismissing.(i3)]))
        @test C[:, .!ismissing.(i3)] == W * onehotbatch(skipmissing(i3), 1:n)
    end
end

@testset "correct post imputing behavior for ngram matrix" begin
    for (m, n, k) in product(fill((2, 5, 10), 3)...)
        W = randn(m, n)
        ψ = randn(m)
        A = PostImputingMatrix(W, ψ)

        S = [randustring(rand(1:100)) for _ in 1:k] |> PooledArray
        B = NGramMatrix(S, 3, 256, n)
        @test A * B ≈ W * Matrix(SparseMatrixCSC(B))
        @inferred A * B
        @test eltype(A * B) ≡ eltype(W)

        S = fill(missing, k) |> PooledArray
        B = NGramMatrix(S, 3, 256, n)
        @test all(isequal(ψ), eachcol(A * B))
        @inferred A * B
        @test eltype(A * B) ≡ eltype(ψ)

        if k > 1
            S = [isodd(i) ? missing : randustring(rand(1:10)) for i in 1:k]
            B = NGramMatrix(S, 3, 256, n)
            C = A * B
            @test all(isequal(ψ), eachcol(C[:, ismissing.(S)]))
            @test C[:, .!ismissing.(S)] ≈ W * NGramMatrix(skipmissing(S) |> collect, 3, 256, n)
            @inferred A * B
            @test eltype(A * B) ≡ promote_type(eltype(W), eltype(ψ))
        end
    end
end

@testset "correct type promotion for post imputing" for A in [
        PostImputingMatrix(rand(1:3, 3, 2), rand(1:3, 3)),
        PostImputingMatrix(rand(3, 2), rand(3)),
        PostImputingMatrix(rand(Float32, 3, 2), rand(Float32, 3))
    ]

    S = [randustring(rand(1:100)) for _ in 1:4]
    B = NGramMatrix(S, 3, 256, 2)
    @test eltype(A * B) ≡ eltype(A)
    S = [isodd(i) ? missing : randustring(rand(1:10)) for i in 1:4] |> PooledArray
    B = NGramMatrix(S, 3, 256, 2)
    @test eltype(A * B) ≡ eltype(A)
    S = fill(missing, 4)
    B = NGramMatrix(S, 3, 256, 2)
    @test eltype(A * B) ≡ eltype(A)

    B = maybehot(1, 1:2)
    @test eltype(A * B) ≡ eltype(A)
    B = maybehot(missing, 1:2)
    @test eltype(A * B) ≡ eltype(A)

    B = maybehotbatch([1, 2], 1:2)
    @test eltype(A * B) ≡ eltype(A)
    B = maybehotbatch([1, missing], 1:2)
    @test eltype(A * B) ≡ eltype(A)
    B = maybehotbatch(fill(missing, 2), 1:2)
    @test eltype(A * B) ≡ eltype(A)
end

@testset "imputing matrix * full vector gradient testing" begin
    for (m, n) in product(fill((2, 5, 10), 2)...)
        b = randn(n)
        W = randn(m, n)

        ψ = randn(n)
        A = PreImputingMatrix(W, ψ)

        @test @gradtest (W, ψ, b) -> PreImputingMatrix(W, ψ) * b

        (dW, dψ), db = gradient(sum ∘ *, A, b)
        @test dW ≈ gradient(W -> sum(W * b), W) |> only
        @test isnothing(dψ)
        @test db ≈ gradient(b -> sum(W * b), b) |> only

        ψ = randn(m)
        A = PostImputingMatrix(W, ψ)

        @test @gradtest (W, ψ, b) -> PostImputingMatrix(W, ψ) * b

        (dW, dψ), db = gradient(sum ∘ *, A, b)
        @test dW ≈ gradient(W -> sum(W * b), W) |> only
        @test isnothing(dψ)
        @test db ≈ gradient(b -> sum(W * b), b) |> only
    end
end

@testset "imputing matrix * full matrix gradient testing" begin
    for (m, n, k) in product(fill((2, 5, 10), 3)...)
        B = randn(n, k)
        W = randn(m, n)

        ψ = randn(n)
        A = PreImputingMatrix(W, ψ)

        @test @gradtest (W, ψ, B) -> PreImputingMatrix(W, ψ) * B

        (dW, dψ), dB = gradient(sum ∘ *, A, B)
        @test dW ≈ gradient(W -> sum(W * B), W) |> only
        @test isnothing(dψ)
        @test dB ≈ gradient(B -> sum(W * B), B) |> only

        ψ = randn(m)
        A = PostImputingMatrix(W, ψ)

        @test @gradtest (W, ψ, B) -> PostImputingMatrix(W, ψ) * B

        (dW, dψ), dB = gradient(sum ∘ *, A, B)
        @test dW ≈ gradient(W -> sum(W * B), W) |> only
        @test isnothing(dψ)
        @test dB ≈ gradient(B -> sum(W * B), B) |> only
    end
end

@testset "post imputing matrix * full maybe hot vector gradient testing" begin
    for (m, n) in product(fill((2, 5, 10), 2)...), i in [rand(1:n) for _ in 1:3]
        W = randn(m, n)
        ψ = randn(m)
        A = PostImputingMatrix(W, ψ)
        b = MaybeHotVector(i, n)

        @test @gradtest (W, ψ) -> PostImputingMatrix(W, ψ) * b

        (dW, dψ), db = gradient(sum ∘ *, A, b)
        @test dW ≈ gradient(W -> sum(W * b), W) |> only
        @test isnothing(dψ)
        @test isnothing(db)
    end
end

@testset "post imputing matrix * full maybe hot matrix gradient testing" begin
    for (m, n, k) in product(fill((2, 5, 10), 3)...), I in [rand(1:n, k) for _ in 1:3]
        W = randn(m, n)
        ψ = randn(m)
        A = PostImputingMatrix(W, ψ)
        B = MaybeHotMatrix(I, n)

        @test @gradtest (W, ψ) -> PostImputingMatrix(W, ψ) * B

        (dW, dψ), dB = gradient(sum ∘ *, A, B)
        @test dW ≈ gradient(W -> sum(W * B), W) |> only
        @test isnothing(dψ)
        @test isnothing(dB)
    end
end

@testset "post imputing matrix * full ngram matrix gradient testing" begin
    for (m, n, k) in product(fill((2, 5, 10), 3)...)
        W = randn(m, n)
        ψ = randn(m)
        A = PostImputingMatrix(W, ψ)
        S = [randustring(rand(1:100)) for _ in 1:k] |> PooledArray
        B = NGramMatrix(S, 3, 256, n)

        @test @gradtest (W, ψ) -> PostImputingMatrix(W, ψ) * B

        (dW, dψ), dB = gradient(sum ∘ *, A, B)
        @test dW ≈ gradient(W -> sum(W * B), W) |> only
        @test isnothing(dψ)
        @test isnothing(dB)
    end
end

@testset "post imputing matrix * empty maybe hot vector gradient testing" begin
    for (m, n) in product(fill((2, 5, 10), 2)...)
        W = randn(m, n)
        ψ = randn(m)
        A = PostImputingMatrix(W, ψ)
        b = MaybeHotVector(missing, n)

        @test @gradtest (W, ψ) -> PostImputingMatrix(W, ψ) * b

        (dW, dψ), db = gradient(sum ∘ *, A, b)
        @test isnothing(dW)
        @test all(dψ .== 1)
        @test isnothing(db)
    end
end

@testset "post imputing matrix * empty maybe hot matrix gradient testing" begin
    for (m, n, k) in product(fill((2, 5, 10), 3)...)
        W = randn(m, n)
        ψ = randn(m)
        A = PostImputingMatrix(W, ψ)
        B = MaybeHotMatrix(fill(missing, k), n)

        @test @gradtest (W, ψ) -> PostImputingMatrix(W, ψ) * B

        (dW, dψ), dB = gradient(sum ∘ *, A, B)
        @test isnothing(dW)
        @test all(dψ .== k)
        @test isnothing(dB)
    end
end

@testset "post imputing matrix * empty ngram matrix gradient testing" begin
    for (m, n, k) in product(fill((2, 5, 10), 3)...)
        W = randn(m, n)
        ψ = randn(m)
        A = PostImputingMatrix(W, ψ)
        S = fill(missing, k)
        B = NGramMatrix(S, 3, 256, n)

        @test @gradtest (W, ψ) -> PostImputingMatrix(W, ψ) * B

        (dW, dψ), dB = gradient(sum ∘ *, A, B)
        @test isnothing(dW)
        @test all(dψ .== k)
        @test isnothing(dB)
    end
end

@testset "post imputing matrix * mixed maybe hot matrix gradient testing" begin
    for (m, n, k) in product(fill((2, 5, 10), 3)...)
        W = randn(m, n)
        ψ = randn(m)
        A = PostImputingMatrix(W, ψ)
        I = [isodd(i) ? missing : rand(1:n) for i in 1:k]
        B = MaybeHotMatrix(I, n)

        @test @gradtest (W, ψ) -> PostImputingMatrix(W, ψ) * B

        (dW, dψ), dB = gradient(sum ∘ *, A, B)
        Bskip = MaybeHotMatrix(skipmissing(I) |> collect, n)
        @test dW ≈ gradient(W -> sum(PostImputingMatrix(W, ψ) * Bskip), W) |> only
        Bmiss = MaybeHotMatrix(filter(ismissing, I) |> collect, n)
        @test dψ ≈ gradient(ψ -> sum(PostImputingMatrix(W, ψ) * Bmiss), ψ) |> only
        @test all(dψ .== count(ismissing, I))
        @test isnothing(dB)
    end
end

@testset "post imputing matrix * mixed ngram matrix gradient testing" begin
    for (m, n, k) in product(fill((2, 5, 10), 3)...)
        W = randn(m, n)
        ψ = randn(m)
        A = PostImputingMatrix(W, ψ)
        S = [isodd(i) ? missing : randustring(rand(1:100)) for i in 1:k] |> PooledArray
        B = NGramMatrix(S, 3, 256, n)

        @test @gradtest (W, ψ) -> PostImputingMatrix(W, ψ) * B

        (dW, dψ), dB = gradient(sum ∘ *, A, B)
        Bskip = NGramMatrix(skipmissing(S) |> collect, 3, 256, n)
        @test dW ≈ gradient(W -> sum(PostImputingMatrix(W, ψ) * Bskip), W) |> only
        Bmiss = NGramMatrix(filter(ismissing, S) |> collect, 3, 256, n)
        @test dψ ≈ gradient(ψ -> sum(PostImputingMatrix(W, ψ) * Bmiss), ψ) |> only
        @test all(dψ .== count(ismissing, S))
        @test isnothing(dB)
    end
end

@testset "pre imputing matrix * empty vector gradient testing" begin
    for (m, n) in product(fill((1, 5, 10), 2)...)
        W = randn(m, n)
        ψ = randn(n)
        A = PreImputingMatrix(W, ψ)
        b = fill(missing, n)

        @test @gradtest (W, ψ) -> PreImputingMatrix(W, ψ) * b

        (dW, dψ), db = gradient(sum ∘ *, A, b)
        @test dW ≈ gradient(W -> sum(W * ψ), W) |> only
        @test dψ ≈ gradient(ψ -> sum(W * ψ), ψ) |> only
        @test isnothing(db)
    end
end

@testset "pre imputing matrix * empty missing matrix gradient testing" begin
    for (m, n, k) in product(fill((1, 5, 10), 3)...)
        W = randn(m, n)
        ψ = randn(n)
        A = PreImputingMatrix(W, ψ)
        B = fill(missing, n, k)

        @test @gradtest (W, ψ) -> PreImputingMatrix(W, ψ) * B

        (dW, dψ), dB = gradient(sum ∘ *, A, B)
        @test dW ≈ gradient(W -> sum(W * repeat(ψ, 1, k)), W) |> only
        @test dψ ≈ gradient(ψ -> sum(W * repeat(ψ, 1, k)), ψ) |> only
        @test isnothing(dB)
    end
end

@testset "pre imputing matrix * mixed vector gradient testing" begin
    for (m, n) in product(fill((1, 5, 10), 2)...)
        W = randn(m, n)
        ψ = randn(n)
        b = Vector{Maybe{Float64}}(randn(n))
        b[rand(eachindex(b), rand(1:n))] .= missing

        @test @gradtest (W, ψ) -> PreImputingMatrix(W, ψ) * b
    end
end

@testset "pre imputing matrix * mixed matrix gradient testing" begin
    for (m, n, k) in product(fill((1, 5, 10), 3)...)
        W = randn(m, n)
        ψ = randn(n)
        B = Matrix{Maybe{Float64}}(randn(n, k))
        B[rand(eachindex(B), rand(1:n*k))] .= missing

        @test @gradtest (W, ψ) -> PreImputingMatrix(W, ψ) * B
    end
end

@testset "Parameters get indeed updated with Flux" begin
    function check(d, X, pW, pψ)
        for opt_rule in [Flux.Descent(), Flux.Adam()]
            dc = deepcopy(d)
            opt_state = Flux.setup(opt_rule, dc)
            Flux.train!((dc, X) -> sum(dc(X)), dc, [(X,)], opt_state)
            @test any(dc.weight.W .≠ d.weight.W) == pW
            @test any(dc.weight.ψ .≠ d.weight.ψ) == pψ
        end
    end

    for (m, n, k) in product(fill((2, 5, 10), 3)...)
        d = preimputing_dense(n, m)

        x = randn(Float32, n)
        check(d, x, true, false)
        x = Vector{Maybe{Float32}}(x)
        x[rand(eachindex(x), rand(1:n-1))] .= missing
        check(d, x, true, true)
        check(d, fill(missing, n), false, true)

        X = randn(Float32, n, k)
        check(d, X, true, false)
        X = Matrix{Maybe{Float32}}(X)
        X[rand(eachindex(X), rand(1:n*k-1))] .= missing
        check(d, X, true, true)
        check(d, fill(missing, n, k), false, true)

        d = postimputing_dense(n, m)

        S = [randustring(rand(1:10)) for _ in 1:k]
        X = NGramMatrix(S, 3, 256, n)
        check(d, X, true, false)
        S = Vector{Maybe{String}}(S)
        S[rand(eachindex(S), rand(1:k-1))] .= missing
        X = NGramMatrix(S, 3, 256, n)
        check(d, X, true, true)
        S = fill(missing, k)
        X = NGramMatrix(S, 3, 256, n)
        check(d, X, false, true)

        x = maybehot(rand(1:n), 1:n)
        check(d, x, true, false)
        x = maybehot(missing, 1:n)
        check(d, x, false, true)

        I = [rand(1:n) for _ in 1:k]
        X = maybehotbatch(I, 1:n)
        check(d, X, true, false)
        I = Vector{Maybe{Integer}}(I)
        I[rand(eachindex(I), rand(1:k-1))] .= missing
        X = maybehotbatch(I, 1:n)
        check(d, X, true, true)
        I = fill(missing, k)
        X = maybehotbatch(I, 1:n)
        check(d, X, false, true)
    end
end

@testset "broadcasting" begin
    function check(A::T, W, ψ) where T
        @test A .+ A == T(W .+ W, ψ .+ ψ)
        @test A .- A == T(W .- W, ψ .- ψ)
        @test A .- 1 == T(W .- 1, ψ .- 1) == -1 .+ A
        @test 2 .* A == T(2 .* W, 2 .* ψ) == A .* 2
        @test A .+ A .+ A == 3 .* A == A .* 3
        A .+= A
        @test A == T(W .+ W, ψ .+ ψ)
        A .-= 2 .* A
        @test A == T(-2 .* W, -2 .* ψ)
        A .*= -1
        @test A == T(2 .* W, 2 .* ψ)
    end

    W = randn(3, 3)
    ψ = randn(3)
    A1 = PreImputingMatrix(W, ψ)
    check(A1, copy(W), copy(ψ))
    A2 = PostImputingMatrix(W, ψ)
    check(A2, copy(W), copy(ψ))

    @test_throws MethodError A1 .+ W
    @test_throws MethodError W .+ A1
    @test_throws MethodError A2 .+ W
    @test_throws MethodError W .+ A2
    @test A1 .+ A2 isa Matrix
    @test A2 .* A1 isa Matrix
end

@testset "imputing Dense construction" begin
    A = preimputing_dense(2, 3)
    @test size(A.weight) == (3, 2)
    @test A.weight isa PreImputingMatrix

    A = postimputing_dense(2, 3)
    @test size(A.weight) == (3, 2)
    @test A.weight isa PostImputingMatrix

    d = Dense(4, 5)
    @test preimputing_dense(d).weight |> size == d.weight |> size
    @test postimputing_dense(d).weight |> size == d.weight |> size
end
