# TODO inferrable tests
#
function _test_imput(W, ob::Vector, b::Vector)
    IM = ImputingMatrix(W, ob)
    @test IM*b == W*ob
end

function _test_imput(W, ob::Vector, t=10, p=0.2)
    for _ in 1:t
        b = [rand() < 0.2 ? missing : x for x in ob]
        idcs = rand([true, false], length(ob))
        _test_imput(W, ob, b)
    end
end

@testset "correct imputing behavior" begin
    _test_imput(randn(3,3), randn(3))
    _test_imput(reshape(1:9 |> collect, (3,3)), rand(1:3, 3))
    _test_imput(randn(3,3), randn(3), fill(missing, 3))
    _test_imput(reshape(1:9 |> collect, (3,3)), rand(1:3, 3), fill(missing, 3))

    W = [1 2; 3 4]
    ψ = [2, 1]
    IM = ImputingMatrix(W, ψ)
    B1 = [4 3; 2 1]
    B2 = [missing missing; missing missing]
    B3 = [missing 3; 1 missing]
    B4 = [3 missing; missing 1]

    @test IM*B1 == W*B1
    @test IM*B2 == [W*ψ W*ψ]
    @test IM*B3 == W*[2 3; 1 1]
    @test IM*B4 == W*[3 2; 1 1]
end

@testset "full vector gradient testing" begin
    for (m, n) in product(fill((1, 2, 5, 10), 2)...)
        W = randn(m, n)
        ψ = randn(n)
        A = ImputingMatrix(W, ψ)
        b = randn(n)
        (dW, dψ), db = gradient(sum ∘ *, A, b)

        @test dW ≈ gradient(W -> sum(ImputingMatrix(W,ψ)*b), W) |> only
        @test dW ≈ gradient(W -> sum(W*b), W) |> only

        @test dψ === gradient(ψ -> sum(ImputingMatrix(W,ψ)*b), ψ) |> only
        @test isnothing(dψ)

        @test db ≈ gradient(b -> sum(ImputingMatrix(W,ψ)*b), b) |> only
        @test db ≈ gradient(b -> sum(W*b), b) |> only
    end
end

@testset "full matrix gradient testing" begin
    for (m, n, k) in product(fill((1, 2, 5, 10), 3)...)
        W = randn(m, n)
        ψ = randn(n)
        A = ImputingMatrix(W, ψ)
        B = randn(n, k)
        (dW, dψ), dB = gradient(sum ∘ *, A, B)

        @test dW ≈ gradient(W -> sum(ImputingMatrix(W,ψ)*B), W) |> only
        @test dW ≈ gradient(W -> sum(W*B), W) |> only

        @test dψ === gradient(ψ -> sum(ImputingMatrix(W,ψ)*B), ψ) |> only
        @test isnothing(dψ)

        @test dB ≈ gradient(B -> sum(ImputingMatrix(W,ψ)*B), B) |> only
        @test dB ≈ gradient(B -> sum(W*B), B) |> only
    end
end

@testset "empty vector gradient testing" begin
    for (m, n) in product(fill((1, 2, 5, 10), 2)...)
        W = randn(m, n)
        ψ = randn(n)
        A = ImputingMatrix(W, ψ)
        b = fill(missing, n)
        (dW, dψ), db = gradient(sum ∘ *, A, b)

        @test dW ≈ gradient(W -> sum(ImputingMatrix(W,ψ)*b), W) |> only
        @test dW ≈ gradient(W -> sum(W*ψ), W) |> only

        @test dψ ≈ gradient(ψ -> sum(ImputingMatrix(W,ψ)*b), ψ) |> only
        @test dψ ≈ gradient(ψ -> sum(W*ψ), ψ) |> only

        @test db === gradient(b -> sum(ImputingMatrix(W,ψ)*b), b) |> only
        @test isnothing(db)
    end
end

@testset "empty matrix gradient testing" begin
    for (m, n, k) in product(fill((1, 2, 5, 10), 3)...)
        W = randn(m, n)
        ψ = randn(n)
        A = ImputingMatrix(W, ψ)
        b = fill(missing, n, k)
        (dW, dψ), db = gradient(sum ∘ *, A, b)

        @test dW ≈ gradient(W -> sum(ImputingMatrix(W,ψ)*b), W) |> only
        @test dW ≈ gradient(W -> sum(W*repeat(ψ, 1, k)), W) |> only

        @test dψ ≈ gradient(ψ -> sum(ImputingMatrix(W,ψ)*b), ψ) |> only
        @test dψ ≈ gradient(ψ -> sum(W*repeat(ψ, 1, k)), ψ) |> only

        @test db === gradient(b -> sum(ImputingMatrix(W,ψ)*b), b) |> only
        @test isnothing(db)
    end
end

@testset "mixed vector gradient testing" begin
    for (m, n) in product(fill((3, 5, 10, 20), 2)...)
        W = randn(m, n)
        ψ = randn(n)
        b = Vector{Union{Float64, Missing}}(randn(n))
        b[rand(eachindex(b), rand(2:n-1))] .= missing

        @test gradtest(W -> ImputingMatrix(W, ψ) * b, W)
        @test gradtest(ψ -> ImputingMatrix(W, ψ) * b, ψ)
        @test gradtest(b -> ImputingMatrix(W, ψ) * b, b)
        @test gradtest((W, ψ, b) -> ImputingMatrix(W, ψ) * b, W, ψ, b)
    end
end

@testset "mixed matrix gradient testing" begin
    for (m, n, k) in product(fill((3, 5, 10, 20), 3)...)
        W = randn(m, n)
        ψ = randn(n)
        B = Matrix{Union{Float64, Missing}}(randn(n, k))
        B[rand(eachindex(B), rand(2:n*k-1))] .= missing

        @test gradtest(W -> ImputingMatrix(W, ψ) * B, W)
        @test gradtest(ψ -> ImputingMatrix(W, ψ) * B, ψ)
        @test gradtest(B -> ImputingMatrix(W, ψ) * B, B)
        @test gradtest((W, ψ, B) -> ImputingMatrix(W, ψ) * B, W, ψ, B)
    end
end
