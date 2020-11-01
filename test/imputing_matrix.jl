# TODO inferrable tests
# TODO (Dense) constructor tests
# test multiplication with MaybeHot and Ngrams
# TODO tests for everything

@testset "cat" begin
    Ws = [randn(3,rand(1:10)) for _ in 1:4]
    ψs = [randn(size(Ws[i], 2)) for i in 1:4]
    As = RowImputingMatrix.(Ws, ψs)
    for is in powerset(1:4), is in permutations(is)
        length(is) > 0 || continue
        @test hcat(As[is]...) == RowImputingMatrix(hcat(Ws[is]...), vcat(ψs[is]...))
        @test_throws ArgumentError vcat(As[is]...)
    end

    Ws = [randn(rand(1:10), 3) for _ in 1:4]
    ψs = [randn(size(Ws[i], 1)) for i in 1:4]
    As = ColImputingMatrix.(Ws, ψs)
    for is in powerset(1:4), is in permutations(is)
        length(is) > 0 || continue
        @test vcat(As[is]...) == ColImputingMatrix(vcat(Ws[is]...), vcat(ψs[is]...))
        @test_throws ArgumentError hcat(As[is]...)
    end
end

@testset "correct row imputing behavior for standard vector (maybe missing)" begin
    function _test_imput(W, ob::Vector, b::Vector)
        A = RowImputingMatrix(W, ob)
        @test A * b == W * ob
    end

    function _test_imput(W, ob::Vector, t=10, p=0.2)
        for _ in 1:t
            b = [rand() < 0.2 ? missing : x for x in ob]
            idcs = rand([true, false], length(ob))
            _test_imput(W, ob, b)
        end
    end
    _test_imput(randn(3,3), randn(3))
    _test_imput(reshape(1:9 |> collect, (3,3)), rand(1:3, 3))
    _test_imput(randn(3,3), randn(3), fill(missing, 3))
    _test_imput(reshape(1:9 |> collect, (3,3)), rand(1:3, 3), fill(missing, 3))
end

@testset "Wrong dimensions" begin
    A = RowImputingMatrix(rand(2,3), rand(3))
    @test_throws DimensionMismatch A * []
    @test_throws DimensionMismatch A * [[]]
    @test_throws DimensionMismatch A * [1, 2]
    @test_throws DimensionMismatch A * [missing, missing]
    @test_throws DimensionMismatch A * [1, 2, 3, 4]
    @test_throws DimensionMismatch A * [1, 2, 3, missing]

    A = ColImputingMatrix(rand(2,3), rand(2))
    @test_throws DimensionMismatch A * MaybeHotVector(2, 2)
    @test_throws DimensionMismatch A * MaybeHotVector(missing, 4)
    @test_throws DimensionMismatch A * MaybeHotMatrix(Int[], 1)
    @test_throws DimensionMismatch A * MaybeHotMatrix([1, 2], 5)
    @test_throws DimensionMismatch A * MaybeHotMatrix([missing, 5, 8], 10)
    @test_throws DimensionMismatch A * MaybeHotMatrix([missing, missing], 2)
end

@testset "correct row imputing behavior for standard matrix (maybe missing)" begin
    W = [1 2; 3 4]
    ψ = [2, 1]
    A = RowImputingMatrix(W, ψ)
    B1 = [4 3; 2 1]
    B2 = [missing missing; missing missing]
    B3 = [missing 3; 1 missing]
    B4 = [3 missing; missing 1]

    @test A * B1 == W * B1
    @test A * B2 == [W * ψ W * ψ]
    @test A * B3 == W * [2 3; 1 1]
    @test A * B4 == W * [3 2; 1 1]
end

@testset "correct col imputing behavior for full standard arrays" begin
    W = [1 2; 3 4]
    ψ = [2, 1]
    A = ColImputingMatrix(W, ψ)
    B = [4 3; 2 1]
    b = [3, 1]
    @test A * B == W * B
    @test A * b == W * b

    B = [missing 3; missing 1]
    b = [3, 1]
    @test isequal(A * B, W * B)
    @test isequal(A * b, W * b)

    W = randn(2,3)
    ψ = randn(2)
    A = ColImputingMatrix(W, ψ)

    B = randn(3, 5)
    b = randn(3)
    @test A * B ≈ W * B
    @test A * b ≈ W * b

    B = fill(missing, 3, 5)
    b = fill(missing, 3)
    @test isequal(A * B, W * B)
    @test isequal(A * b, W * b)
end

@testset "correct col imputing behavior for maybe hot vector" begin
    for (m, n) in product(fill((1, 5, 10, 20), 2)...)
        W = randn(m, n)
        ψ = randn(m)
        A = ColImputingMatrix(W, ψ)
        i = rand(1:n)
        b = MaybeHotVector(i, n)
        @test A * b == W * onehot(b)
        b = MaybeHotVector(missing, n)
        @test A * b == ψ
    end
end

@testset "correct col imputing behavior for maybe hot matrix" begin
    for (m, n, k) in product(fill((1, 5, 10, 20), 3)...)
        W = randn(m, n)
        ψ = randn(m)
        A = ColImputingMatrix(W, ψ)
        i1 = rand(1:n, k)
        i2 = fill(missing, k)
        i3 = rand(vcat(missing, 1:n), k)
        B1 = MaybeHotMatrix(i1, n)
        B2 = MaybeHotMatrix(i2, n)
        B3 = MaybeHotMatrix(i3, n)
        @test A * B1 == W * onehotbatch(B1)
        @test all(isequal(ψ), eachcol(A * B2))
        C = A * B3
        @test all(isequal(ψ), eachcol(C[:, ismissing.(i3)]))
        @test C[:, .!ismissing.(i3)] == W * onehotbatch(skipmissing(i3), 1:n)
    end
end

@testset "imputing matrix * full vector gradient testing" begin
    for (m, n) in product(fill((1, 5, 10, 20), 2)...)
        b = randn(n)
        W = randn(m, n)

        ψ = randn(n)
        A = RowImputingMatrix(W, ψ)
        (dW, dψ), db = gradient(sum ∘ *, A, b)

        @test gradtest((W, ψ) -> sum(RowImputingMatrix(W, ψ) * b), W, ψ)

        @test dW ≈ gradient(W -> sum(RowImputingMatrix(W, ψ) * b), W) |> only
        @test dW ≈ gradient(W -> sum(W * b), W) |> only
        @test gradtest(W -> sum(RowImputingMatrix(W, ψ) * b), W)

        @test dψ === gradient(ψ -> sum(RowImputingMatrix(W, ψ) * b), ψ) |> only
        @test isnothing(dψ)
        @test gradtest(ψ -> sum(RowImputingMatrix(W, ψ) * b), ψ)

        @test db ≈ gradient(b -> sum(RowImputingMatrix(W, ψ) * b), b) |> only
        @test db ≈ gradient(b -> sum(W * b), b) |> only

        ψ = randn(m)
        A = ColImputingMatrix(W, ψ)
        (dW, dψ), db = gradient(sum ∘ *, A, b)

        @test gradtest((W, ψ) -> sum(RowImputingMatrix(W, ψ) * b), W, ψ)

        @test dW ≈ gradient(W -> sum(ColImputingMatrix(W, ψ) * b), W) |> only
        @test dW ≈ gradient(W -> sum(W * b), W) |> only
        @test gradtest(W -> sum(ColImputingMatrix(W, ψ) * b), W)

        @test dψ === gradient(ψ -> sum(ColImputingMatrix(W, ψ) * b), ψ) |> only
        @test isnothing(dψ)
        @test gradtest(ψ -> sum(ColImputingMatrix(W, ψ) * b), ψ)

        @test db ≈ gradient(b -> sum(ColImputingMatrix(W, ψ) * b), b) |> only
        @test db ≈ gradient(b -> sum(W * b), b) |> only
    end
end

@testset "imputing matrix * full matrix gradient testing" begin
    for (m, n, k) in product(fill((1, 5, 10, 20), 3)...)
        B = randn(n, k)
        W = randn(m, n)

        ψ = randn(n)
        A = RowImputingMatrix(W, ψ)
        (dW, dψ), dB = gradient(sum ∘ *, A, B)

        @test gradtest((W, ψ) -> sum(RowImputingMatrix(W, ψ) * B), W, ψ)

        @test dW ≈ gradient(W -> sum(RowImputingMatrix(W, ψ) * B), W) |> only
        @test dW ≈ gradient(W -> sum(W * B), W) |> only
        @test gradtest(W -> sum(RowImputingMatrix(W, ψ) * B), W)

        @test dψ === gradient(ψ -> sum(RowImputingMatrix(W, ψ) * B), ψ) |> only
        @test isnothing(dψ)
        @test gradtest(ψ -> sum(RowImputingMatrix(W, ψ) * B), ψ)

        @test dB ≈ gradient(B -> sum(RowImputingMatrix(W, ψ) * B), B) |> only
        @test dB ≈ gradient(B -> sum(W * B), B) |> only

        ψ = randn(m)
        A = ColImputingMatrix(W, ψ)
        (dW, dψ), dB = gradient(sum ∘ *, A, B)

        @test gradtest((W, ψ) -> sum(ColImputingMatrix(W, ψ) * B), W, ψ)

        @test dW ≈ gradient(W -> sum(ColImputingMatrix(W, ψ) * B), W) |> only
        @test dW ≈ gradient(W -> sum(W * B), W) |> only
        @test gradtest(W -> sum(ColImputingMatrix(W, ψ) * B), W)

        @test dψ === gradient(ψ -> sum(ColImputingMatrix(W, ψ) * B), ψ) |> only
        @test isnothing(dψ)
        @test gradtest(ψ -> sum(ColImputingMatrix(W, ψ) * B), ψ)

        @test dB ≈ gradient(B -> sum(ColImputingMatrix(W, ψ) * B), B) |> only
        @test dB ≈ gradient(B -> sum(W * B), B) |> only
    end
end

@testset "col imputing matrix * full maybe hot vector gradient testing" begin
    for (m, n) in product(fill((1, 5, 10, 20), 2)...), i in [rand(1:n) for _ in 1:3]
        W = randn(m, n)
        ψ = randn(m)
        A = ColImputingMatrix(W, ψ)
        b = MaybeHotVector(i, n)

        (dW, dψ), db = gradient(sum ∘ *, A, b)

        @test gradtest((W, ψ) -> sum(ColImputingMatrix(W, ψ) * b), W, ψ)

        @test dW ≈ gradient(W -> sum(ColImputingMatrix(W, ψ) * b), W) |> only
        @test dW ≈ gradient(W -> sum(W * b), W) |> only
        @test gradtest(W -> sum(ColImputingMatrix(W, ψ) * b), W)

        @test dψ === gradient(ψ -> sum(ColImputingMatrix(W, ψ) * b), ψ) |> only
        @test isnothing(dψ)
        @test gradtest(ψ -> sum(ColImputingMatrix(W, ψ) * b), ψ)

        @test db === gradient(b -> sum(ColImputingMatrix(W, ψ) * b), b) |> only
        @test isnothing(db)
    end
end

@testset "col imputing matrix * full maybe hot matrix gradient testing" begin
    for (m, n, k) in product(fill((1, 5, 10, 20), 3)...), I in [rand(1:n, k) for _ in 1:3]
        W = randn(m, n)
        ψ = randn(m)
        A = ColImputingMatrix(W, ψ)
        B = MaybeHotMatrix(I, n)

        (dW, dψ), dB = gradient(sum ∘ *, A, B)

        @test gradtest((W, ψ) -> sum(ColImputingMatrix(W, ψ) * B), W, ψ)

        @test dW ≈ gradient(W -> sum(ColImputingMatrix(W, ψ) * B), W) |> only
        @test dW ≈ gradient(W -> sum(W * B), W) |> only
        @test gradtest(W -> sum(ColImputingMatrix(W, ψ) * B), W)

        @test dψ === gradient(ψ -> sum(ColImputingMatrix(W, ψ) * B), ψ) |> only
        @test isnothing(dψ)
        @test gradtest(ψ -> sum(ColImputingMatrix(W, ψ) * B), ψ)

        @test dB === gradient(B -> sum(ColImputingMatrix(W, ψ) * B), B) |> only
        @test isnothing(dB)
    end
end

@testset "col imputing matrix * empty maybe hot vector gradient testing" begin
    for (m, n) in product(fill((1, 5, 10, 20), 2)...)
        W = randn(m, n)
        ψ = randn(m)
        A = ColImputingMatrix(W, ψ)
        b = MaybeHotVector(missing, n)

        (dW, dψ), db = gradient(sum ∘ *, A, b)

        @test gradtest((W, ψ) -> sum(ColImputingMatrix(W, ψ) * b), W, ψ)

        @test dW === gradient(W -> sum(ColImputingMatrix(W, ψ) * b), W) |> only
        @test gradtest(W -> sum(ColImputingMatrix(W, ψ) * b), W)
        @test isnothing(dW)

        @test dψ ≈ gradient(ψ -> sum(ColImputingMatrix(W, ψ) * b), ψ) |> only
        @test gradtest(ψ -> sum(ColImputingMatrix(W, ψ) * b), ψ)
        @test all(dψ .== 1)

        @test db === gradient(b -> sum(ColImputingMatrix(W, ψ) * b), b) |> only
        @test isnothing(db)
    end
end

@testset "col imputing matrix * empty maybe hot matrix gradient testing" begin
    for (m, n, k) in product(fill((1, 5, 10, 20), 3)...)
        W = randn(m, n)
        ψ = randn(m)
        A = ColImputingMatrix(W, ψ)
        B = MaybeHotMatrix(fill(missing, k), n)

        (dW, dψ), dB = gradient(sum ∘ *, A, B)

        @test gradtest((W, ψ) -> sum(ColImputingMatrix(W, ψ) * B), W, ψ)

        @test dW === gradient(W -> sum(ColImputingMatrix(W, ψ) * B), W) |> only
        @test gradtest(W -> sum(ColImputingMatrix(W, ψ) * B), W)
        @test isnothing(dW)

        @test dψ ≈ gradient(ψ -> sum(ColImputingMatrix(W, ψ) * B), ψ) |> only
        @test gradtest(ψ -> sum(ColImputingMatrix(W, ψ) * B), ψ)
        @test all(dψ .== k)

        @test dB === gradient(B -> sum(ColImputingMatrix(W, ψ) * B), B) |> only
        @test isnothing(dB)
    end
end

@testset "col imputing matrix * mixed maybe hot matrix gradient testing" begin
    for (m, n, k) in product(fill((1, 5, 10, 20), 3)...), I in [rand(vcat(fill(missing, 1:n), 1:n), k) for _ in 1:3]
        W = randn(m, n)
        ψ = randn(m)
        A = ColImputingMatrix(W, ψ)
        B = MaybeHotMatrix(I, n)

        (dW, dψ), dB = gradient(sum ∘ *, A, B)

        @test gradtest((W, ψ) -> sum(ColImputingMatrix(W, ψ) * B), W, ψ)

        @test dW ≈ gradient(W -> sum(ColImputingMatrix(W, ψ) * B), W) |> only
        Bskip = MaybeHotMatrix(skipmissing(I) |> collect, n)
        @test dW ≈ gradient(W -> sum(ColImputingMatrix(W, ψ) * Bskip), W) |> only
        @test gradtest(W -> sum(ColImputingMatrix(W, ψ) * B), W)

        @test dψ ≈ gradient(ψ -> sum(ColImputingMatrix(W, ψ) * B), ψ) |> only
        Bmiss = MaybeHotMatrix(filter(ismissing, I) |> collect, n)
        @test dψ ≈ gradient(ψ -> sum(ColImputingMatrix(W, ψ) * Bmiss), ψ) |> only
        @test gradtest(ψ -> sum(ColImputingMatrix(W, ψ) * B), ψ)
        @test all(dψ .== count(ismissing, I))

        @test dB === gradient(B -> sum(ColImputingMatrix(W, ψ) * B), B) |> only
        @test isnothing(dB)
    end
end

@testset "row imputing matrix * empty vector gradient testing" begin
    for (m, n) in product(fill((1, 5, 10, 20), 2)...)
        W = randn(m, n)
        ψ = randn(n)
        A = RowImputingMatrix(W, ψ)
        b = fill(missing, n)
        (dW, dψ), db = gradient(sum ∘ *, A, b)

        @test gradtest((W, ψ) -> sum(RowImputingMatrix(W, ψ) * b), W, ψ)

        @test dW ≈ gradient(W -> sum(RowImputingMatrix(W, ψ) * b), W) |> only
        @test dW ≈ gradient(W -> sum(W * ψ), W) |> only
        @test gradtest(W -> sum(RowImputingMatrix(W, ψ) * b), W)

        @test dψ ≈ gradient(ψ -> sum(RowImputingMatrix(W, ψ) * b), ψ) |> only
        @test dψ ≈ gradient(ψ -> sum(W * ψ), ψ) |> only
        @test gradtest(ψ -> sum(RowImputingMatrix(W, ψ) * b), ψ)

        @test db === gradient(b -> sum(RowImputingMatrix(W, ψ) * b), b) |> only
        @test isnothing(db)
    end
end

@testset "row imputing matrix * empty missing matrix gradient testing" begin
    for (m, n, k) in product(fill((1, 5, 10, 20), 3)...)
        W = randn(m, n)
        ψ = randn(n)
        A = RowImputingMatrix(W, ψ)
        B = fill(missing, n, k)
        (dW, dψ), dB = gradient(sum ∘ *, A, B)

        @test gradtest((W, ψ) -> sum(RowImputingMatrix(W, ψ) * B), W, ψ)

        @test dW ≈ gradient(W -> sum(RowImputingMatrix(W, ψ) * B), W) |> only
        @test dW ≈ gradient(W -> sum(W * repeat(ψ, 1, k)), W) |> only
        @test gradtest(W -> sum(RowImputingMatrix(W, ψ) * B), W)

        @test dψ ≈ gradient(ψ -> sum(RowImputingMatrix(W, ψ) * B), ψ) |> only
        @test dψ ≈ gradient(ψ -> sum(W * repeat(ψ, 1, k)), ψ) |> only
        @test gradtest(ψ -> sum(RowImputingMatrix(W, ψ) * B), ψ)

        @test dB === gradient(B -> sum(RowImputingMatrix(W, ψ) * B), B) |> only
        @test isnothing(dB)
    end
end

@testset "row imputing matrix * mixed vector gradient testing" begin
    for (m, n) in product(fill((1, 5, 10, 20), 2)...)
        W = randn(m, n)
        ψ = randn(n)
        b = Vector{Union{Float64, Missing}}(randn(n))
        b[rand(eachindex(b), rand(1:n))] .= missing

        @test gradtest(W -> RowImputingMatrix(W, ψ) * b, W)
        @test gradtest(ψ -> RowImputingMatrix(W, ψ) * b, ψ)
        @test gradtest(b -> RowImputingMatrix(W, ψ) * b, b)
        @test gradtest((W, ψ, b) -> RowImputingMatrix(W, ψ) * b, W, ψ, b)
    end
end

@testset "row imputing matrix * mixed matrix gradient testing" begin
    for (m, n, k) in product(fill((1, 5, 10, 20), 3)...)
        W = randn(m, n)
        ψ = randn(n)
        B = Matrix{Union{Float64, Missing}}(randn(n, k))
        B[rand(eachindex(B), rand(1:n*k))] .= missing

        @test gradtest(W -> RowImputingMatrix(W, ψ) * B, W)
        @test gradtest(ψ -> RowImputingMatrix(W, ψ) * B, ψ)
        @test gradtest(B -> RowImputingMatrix(W, ψ) * B, B)
        @test gradtest((W, ψ, B) -> RowImputingMatrix(W, ψ) * B, W, ψ, B)
    end
end
