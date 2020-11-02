@testset "maybe hot vector io" begin
    x = MaybeHotVector(1, 3)
    @test repr(x) == "MaybeHotVector(i = 1, l = 3)"
    @test repr(x; context=:compact => true) == "3x1 MaybeHotVector"
    @test repr(MIME("text/plain"), x) ==
        "3-element MaybeHotVector{Int64,Int64,Bool}:\n 1\n 0\n 0"

    x = MaybeHotVector(missing, 4)
    @test repr(x) == "MaybeHotVector(i = missing, l = 4)"
    @test repr(x; context=:compact => true) == "4x1 MaybeHotVector"
    @test repr(MIME("text/plain"), x) ==
        "4-element MaybeHotVector{Missing,Int64,Missing}:\n missing\n missing\n missing\n missing"
end

@testset "maybe hot matrix io" begin
    X = MaybeHotMatrix([1, 3], 3)
    @test repr(X) == "MaybeHotMatrix(I = [1, 3], l = 3)"
    @test repr(X; context=:compact => true) == "3x2 MaybeHotMatrix"
    @test repr(MIME("text/plain"), X) ==
        "3×2 MaybeHotMatrix{Int64,Array{Int64,1},Int64,Bool}:\n 1  0\n 0  0\n 0  1"

    X = MaybeHotMatrix([missing, 2], 2)
    @test repr(X) == "MaybeHotMatrix(I = Union{Missing, Int64}[missing, 2], l = 2)"
    @test repr(X; context=:compact => true) == "2x2 MaybeHotMatrix"
    @test repr(MIME("text/plain"), X) ==
        "2×2 MaybeHotMatrix{Union{Missing, Int64},Array{Union{Missing, Int64},1},Int64,Union{Missing, Bool}}:" *
         "\n missing  false\n missing   true"

    X = MaybeHotMatrix([missing], 4)
    @test repr(X) == "MaybeHotMatrix(I = [missing], l = 4)"
    @test repr(X; context=:compact => true) == "4x1 MaybeHotMatrix"
    @test repr(MIME("text/plain"), X) ==
        "4×1 MaybeHotMatrix{Missing,Array{Missing,1},Int64,Missing}:" *
        "\n missing\n missing\n missing\n missing"
end

@testset "row imputing matrix io" begin
    X = RowImputingMatrix(reshape(1:8, 4, 2) |> Matrix)
    @test repr(X) == "RowImputingMatrix(W = [1 5; 2 6; 3 7; 4 8], ψ = [0, 0])"
    @test repr(X; context=:compact => true) == "4x2 RowImputingMatrix"
    @test repr(MIME("text/plain"), X) == "4×2 RowImputingMatrix{Int64,Array{Int64,1},Array{Int64,2}}:" *
        "\nW:\n 1  5\n 2  6\n 3  7\n 4  8\n\nψ:\n 0  0"
end

@testset "col imputing matrix io" begin
    X = ColImputingMatrix(reshape(1:6, 2, 3) |> Matrix)
    @test repr(X) == "ColImputingMatrix(W = [1 3 5; 2 4 6], ψ = [0, 0])"
    @test repr(X; context=:compact => true) == "2x3 ColImputingMatrix"
    @test repr(MIME("text/plain"), X) == "2×3 ColImputingMatrix{Int64,Array{Int64,1},Array{Int64,2}}:" *
        "\nW:\n 1  3  5\n 2  4  6\n\nψ:\n 0\n 0"
end

@testset "ngram matrix io" begin
    X = NGramMatrix(["a", "b", "c"])
    @test repr(X) == "NGramMatrix(s = [\"a\", \"b\", \"c\"], n = 3, b = 256, m = 2053)"
    @test repr(X; context=:compact => true) == "2053x3 NGramMatrix"
    @test repr(MIME("text/plain"), X) == "2053×3 NGramMatrix{String}:\n \"a\"\n \"b\"\n \"c\""
end
