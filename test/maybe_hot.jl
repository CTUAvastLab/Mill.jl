@testset "attributes" begin
    l = 10
    I = [1, missing, 3, missing, 5]

    for i in I
        mhv = MaybeHotVector(i, l)
        @test size(mhv) == (l,)
        @test length(mhv) == l
    end

    mhm = MaybeHotMatrix(I, l)
    @test size(mhm) == (l, length(I))
    @test length(mhm) == l * length(I)
end

@testset "type construction" begin
    @test MaybeHotVector(1, 10) isa AbstractVector{Bool}
    @test MaybeHotVector(missing, 10) isa AbstractVector{Missing}
    @test MaybeHotMatrix([1, 2], 10) isa AbstractMatrix{Bool}
    @test MaybeHotMatrix([missing, missing], 10) isa AbstractMatrix{Missing}
    @test MaybeHotMatrix([1, missing], 10) isa AbstractMatrix{Union{Bool, Missing}}
end

@testset "comparisons" begin
    @test MaybeHotVector(2, 10) == MaybeHotVector(2, 10)
    @test isequal(MaybeHotVector(1, 10), MaybeHotVector(1, 10))

    @test isequal(MaybeHotVector(missing, 10), MaybeHotVector(missing, 10))

    @test MaybeHotMatrix([7,2], 10) == MaybeHotMatrix([7,2], 10)
    @test isequal(MaybeHotMatrix([1,3,9], 10), MaybeHotMatrix([1,3,9], 10))

    @test isequal(MaybeHotMatrix([1,2,missing], 10), MaybeHotMatrix([1,2,missing], 10))
    @test !isequal(MaybeHotMatrix([1,2,missing,3], 10) == MaybeHotMatrix([1,2,missing,4], 10))
end

@testset "hcat" begin
    l = 10
    I = [1, missing, 3, missing, 5]
    mhm = MaybeHotMatrix(I, l)
    mhm2 = MaybeHotMatrix([1,5,3], l)
    mhm3 = MaybeHotMatrix(fill(missing, 4), l)
    mhvs = [MaybeHotVector.(I, l)...]

    @test all(mhv -> areequal(hcat(mhv), reduce(hcat, [mhv]), MaybeHotMatrix(mhv)),
              mhvs)

    @test all(mhm -> areequal(hcat(mhm), reduce(hcat, [mhm]), mhm), [mhm, mhm2, mhm3])

    @test areequal(hcat(mhvs...), reduce(hcat, mhvs), mhm)

    @test areequal(hcat(mhm, mhm),
                   reduce(hcat, [mhm, mhm]),
                   hcat(mhvs..., mhvs...),
                   reduce(hcat, [mhvs..., mhvs...]),
                   MaybeHotMatrix(vcat(I, I), l))

    @test areequal(hcat(mhm, mhm2), reduce(hcat, [mhm, mhm2]), MaybeHotMatrix(vcat(mhm.I, mhm2.I), l))
    @test areequal(hcat(mhm, mhm3), reduce(hcat, [mhm, mhm3]), MaybeHotMatrix(vcat(mhm.I, mhm3.I), l))
    @test areequal(hcat(mhm3, mhm2), reduce(hcat, [mhm3, mhm2]), MaybeHotMatrix(vcat(mhm3.I, mhm2.I), l))

    @inferred hcat(mhm)
    @inferred reduce(hcat, [mhm])
    @inferred hcat(mhm2)
    @inferred reduce(hcat, [mhm2])
    @inferred hcat(mhm3)
    @inferred reduce(hcat, [mhm3])
    @inferred hcat(mhm, mhm2)
    @inferred reduce(hcat, [mhm, mhm2])
    @inferred hcat(mhm, mhm3)
    @inferred reduce(hcat, [mhm, mhm3])
    @inferred hcat(mhm, mhm2, mhm3)
    @inferred reduce(hcat, [mhm, mhm2, mhm3])
    @inferred hcat(mhvs...)
    @inferred reduce(hcat, mhvs)

    @test_throws DimensionMismatch hcat(MaybeHotVector.([1, 2], [l, l+1])...)
    @test_throws DimensionMismatch hcat(MaybeHotMatrix.([[1], [2, 3]], [l, l+1])...)
    @test_throws DimensionMismatch reduce(hcat, MaybeHotVector.([1, 2], [l, l+1]))
    @test_throws DimensionMismatch reduce(hcat, MaybeHotMatrix.([[1], [2, 3]], [l, l+1]))
end

@testset "indexing" begin
    l = 10
    I = [1, missing, 3, missing, 5]

    for i in I
        mhv = MaybeHotVector(i, l)
        if ismissing(i)
            @test all(isequal.(missing, mhv))
        else
            @test Vector(mhv) == onehot(i, 1:l)
        end
        @test_throws BoundsError mhv[0]
        @test_throws BoundsError mhv[l+1]
        @test all(isequal.(mhv[:], mhv))
    end

    mhm = MaybeHotMatrix(I, l)
    m = Matrix(mhm)
    for (k,i) in I |> enumerate
        if ismissing(i)
            @test all(isequal.(missing, m[:, k]))
            @test all(isequal.(missing, mhm[:, k]))
        else
            @test mhm[:, k] == m[:, k] == onehot(i, 1:l)
        end
    end
    @test isequal(mhm[[1,2,7], 3], m[[1,2,7], 3])
    @test isequal(mhm[CartesianIndex(2, 4)], m[2,4])
    for k in eachindex(I)
        @test isequal(mhm[:, k], MaybeHotVector(I[k], l))
    end
    @test isequal(mhm, mhm[:, eachindex(I)])
    @test isequal(mhm, mhm[:, eachindex(I) |> collect])
    @test isequal(mhm, mhm[:, :])
    @test isequal(mhm, hcat(MaybeHotVector.(I, l)...))
    @test isequal(mhm[:, [1,2,5]], hcat(MaybeHotVector.(I[[1,2,5]], l)...))

    @test_throws BoundsError mhm[0, 1]
    @test_throws BoundsError mhm[2, -1]
    @test_throws BoundsError mhm[CartesianIndex()]
    @test_throws BoundsError mhm[CartesianIndex(1)]
end

@testset "multiplication" begin
    W = rand(10, 10)
    x1 = MaybeHotVector(1, 10)
    x2 = MaybeHotVector(8, 10)
    x3 = MaybeHotVector(missing, 10)
    X1 = MaybeHotMatrix([7, 10], 10)
    X2 = MaybeHotMatrix([missing, missing], 10)
    X3 = MaybeHotMatrix([3, missing, 1, missing], 10)

    @test isequal(W * x1, W * Vector(x1))
    @test isequal(W * x2, W * Vector(x2))
    @test isequal(W * x3, W * Vector(x3))
    @test isequal(W * X1, W * Matrix(X1))
    @test isequal(W * X2, W * Matrix(X2))
    @test isequal(W * X3, W * Matrix(X3))

    @test eltype(W * x1) === eltype(W * x2) === eltype(W * X1) === eltype(W)
    @test eltype(W * x3) === eltype(W * X2) === Missing
    @test eltype(W * X3) === Union{Missing, eltype(W)}

    @inferred W * x1
    @inferred W * x2
    @inferred W * x3
    @inferred W * X1
    @inferred W * X2
    @inferred W * X3

    @test_throws DimensionMismatch W * MaybeHotVector(1, 5)
    @test_throws DimensionMismatch W * MaybeHotVector(missing, 3)
    @test_throws DimensionMismatch W * MaybeHotMatrix([1, 2], 9)
    @test_throws DimensionMismatch W * MaybeHotMatrix([1, missing, 2], 9)
    @test_throws DimensionMismatch W * MaybeHotMatrix([missing, missing], 9)
end

@testset "equality" begin
    mhv1 = MaybeHotVector(1, 10)
    mhv2 = MaybeHotVector(1, 10)
    mhv3 = MaybeHotVector(1, 11)
    mhv4 = MaybeHotVector(missing, 11)
    mhv5 = MaybeHotVector(2, 10)
    @test mhv1 == mhv2
    @test mhv1 != mhv3
    @test !isequal(mhv1, mhv4)
    @test mhv1 != mhv5

    mhm1 = MaybeHotMatrix([1,2], 10)
    mhm2 = MaybeHotMatrix([1,2], 10)
    mhm3 = MaybeHotMatrix([1], 10)
    mhm4 = MaybeHotMatrix([missing], 10)
    mhm5 = MaybeHotMatrix([1,2], 11)
    @test mhm1 == mhm2
    @test mhm1 != mhm3
    @test !isequal(mhv1, mhv4)
    @test mhm1 != mhm5
end

@testset "onehot and onehotbatch" begin
    i = 1
    b = MaybeHotVector(i, 10)
    @test onehot(b) == onehot(i, 1:length(b))
    b = MaybeHotVector(missing, 10)
    @test_throws MethodError onehot(b)
    I = [3, 1, 2]
    B = MaybeHotMatrix(I, 10)
    @test onehotbatch(B) == onehotbatch(I, 1:size(B, 1))
    B = MaybeHotMatrix([3, missing, 2], 10)
    @test_throws MethodError onehotbatch(B)
end

@testset "AbstractMatrix * MaybeHotVector{<:Integer} gradtest" begin
    # for MaybeHot types with missing elements, it doesn't make sense to compute gradient
    for (m, n) in product(fill((1, 5, 10, 20), 2)...), i in [rand(1:n) for _ in 1:3]
        A = randn(m, n)
        b = MaybeHotVector(i, n)

        @test gradtest(A -> A * b, A)

        dA, db = gradient(sum ∘ *, A, b)
        @test dA ≈ gradient(A -> sum(A * onehot(b)), A) |> only
        @test isnothing(db)
    end
end

@testset "AbstractMatrix * MaybeHotMatrix{<:Integer} gradtest" begin
    # for MaybeHot types with missing elements, it doesn't make sense to compute gradient
    for (m, n, k) in product(fill((1, 5, 10, 20), 3)...), I in [rand(1:n, k) for _ in 1:3]
        A = randn(m, n)
        B = MaybeHotMatrix(I, n)

        @test gradtest(A -> A * B, A)

        dA, dB = gradient(sum ∘ *, A, B)
        @test dA ≈ gradient(A -> sum(A * onehotbatch(B)), A) |> only
        @test isnothing(dB)
    end
end

@testset "maybehot" begin
    @test_throws ArgumentError maybehot(0, 1:3)
    @test_throws ArgumentError maybehot(4, 1:3)
    @test_throws ArgumentError maybehot("a", [1, 2])

    @test ismissing(maybehot(missing, 1:3).i)
    @test maybehot(2, 1:3).i == 2

    @test maybehot(missing, 1:3) isa AbstractVector{Missing}
    @test maybehot(2, 1:3) isa AbstractVector{Bool}

    @test maybehot(missing, 1:3) isa MaybeHotVector{Missing}
    @test maybehot(2, 1:3) isa MaybeHotVector{Int}
end

@testset "maybehotbatch" begin
    @test_throws ArgumentError maybehotbatch([1, 2, 3, 0], 1:3)
    @test_throws ArgumentError maybehotbatch([4], 1:3)
    @test_throws ArgumentError maybehotbatch([2, "a"], [1, 2])

    mhm = maybehotbatch([missing, missing], 1:3)

    @test all(isequal.(mhm.I, [missing, missing]))
    @test mhm isa AbstractMatrix{Missing}
    @test mhm isa MaybeHotMatrix{Missing}

    mhm = maybehotbatch([3, 1], 1:3)

    @test mhm.I == [3, 1]
    @test mhm isa AbstractMatrix{Bool}
    @test mhm isa MaybeHotMatrix{Int}

    mhm = maybehotbatch([1, missing], 1:3)

    @test all(isequal.(mhm.I, [1, missing]))
    @test mhm isa AbstractMatrix{Union{Bool, Missing}}
    @test mhm isa MaybeHotMatrix{Union{Int, Missing}}
end
