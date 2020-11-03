@testset "data and model node io" begin
    an = ArrayNode(ones(2,5))
    @test repr(an) == "ArrayNode(2×5 Array{Float64,2}) with 5 obs"
    @test repr(an; context=:compact => true) == "ArrayNode"
    @test repr(MIME("text/plain"), an) == repr(an)

    anm = reflectinmodel(an)
    @test repr(anm) == "ArrayModel(Dense(2, 10))"
    @test repr(anm; context=:compact => true) == "ArrayModel"
    @test repr(MIME("text/plain"), anm) == repr(anm)

    bn = BagNode(an, bags([1:2, 3:5]))
    @test repr(bn) == "BagNode with 2 obs"
    @test repr(bn; context=:compact => true) == "BagNode"
    @test repr(MIME("text/plain"), bn) == 
        """
        BagNode with 2 obs
          └── ArrayNode(2×5 Array{Float64,2}) with 5 obs"""

    bnm = reflectinmodel(bn)
    @test repr(bnm) == "BagModel ↦ ⟨SegmentedMean(10)⟩ ↦ ArrayModel(Dense(11, 10))"
    @test repr(bnm; context=:compact => true) == "BagModel"
    @test repr(MIME("text/plain"), bnm) == 
        """
        BagModel ↦ ⟨SegmentedMean(10)⟩ ↦ ArrayModel(Dense(11, 10))
          └── ArrayModel(Dense(2, 10))"""

    wbn = WeightedBagNode(an, bags([1:2, 3:5]), rand(5) |> f32)
    @test repr(wbn) == "WeightedBagNode with 2 obs"
    @test repr(wbn; context=:compact => true) == "WeightedBagNode"
    @test repr(MIME("text/plain"), wbn) == 
        """
        WeightedBagNode with 2 obs
          └── ArrayNode(2×5 Array{Float64,2}) with 5 obs"""

    wbnm = reflectinmodel(wbn)
    @test repr(wbnm) == "BagModel ↦ ⟨SegmentedMean(10)⟩ ↦ ArrayModel(Dense(11, 10))"
    @test repr(wbnm; context=:compact => true) == "BagModel"
    @test repr(MIME("text/plain"), wbnm) == 
        """
        BagModel ↦ ⟨SegmentedMean(10)⟩ ↦ ArrayModel(Dense(11, 10))
          └── ArrayModel(Dense(2, 10))"""

    pn = ProductNode((;bn, wbn))
    @test repr(pn) == "ProductNode with 2 obs"
    @test repr(pn; context=:compact => true) == "ProductNode"
    @test repr(MIME("text/plain"), pn) ==
        """
        ProductNode with 2 obs
          ├─── bn: BagNode with 2 obs
          │          └── ArrayNode(2×5 Array{Float64,2}) with 5 obs
          └── wbn: WeightedBagNode with 2 obs
                     └── ArrayNode(2×5 Array{Float64,2}) with 5 obs"""

    pnm = reflectinmodel(pn)
    @test repr(pnm) == "ProductModel ↦ ArrayModel(Dense(20, 10))"
    @test repr(pnm; context=:compact => true) == "ProductModel"
    @test repr(MIME("text/plain"), pnm) ==
        """
        ProductModel ↦ ArrayModel(Dense(20, 10))
          ├─── bn: BagModel ↦ ⟨SegmentedMean(10)⟩ ↦ ArrayModel(Dense(11, 10))
          │          └── ArrayModel(Dense(2, 10))
          └── wbn: BagModel ↦ ⟨SegmentedMean(10)⟩ ↦ ArrayModel(Dense(11, 10))
                     └── ArrayModel(Dense(2, 10))"""

    ln = LazyNode{:Sentence}(["a", "b", "c", "d"])
    @test repr(ln) == "LazyNode{Sentence} with 4 obs"
    @test repr(ln; context=:compact => true) == "LazyNode"
    @test repr(MIME("text/plain"), ln) == repr(ln)

    lnm = reflectinmodel(ln)
    @test repr(lnm) == "LazyModel{Sentence}"
    @test repr(lnm; context=:compact => true) == "LazyModel"
    @test repr(MIME("text/plain"), lnm) == 
        """
        LazyModel{Sentence}
          └── BagModel ↦ ⟨SegmentedMean(10)⟩ ↦ ArrayModel(Dense(11, 10))
                └── ArrayModel(Dense(2053, 10))"""
end

@testset "aggregation io" begin
    a = SegmentedMax(3)
    @test repr(a) == "⟨SegmentedMax(3)⟩"
    @test repr(a; context=:compact => true) == "Aggregation(3)"
    @test repr(MIME("text/plain"), a) == "Aggregation{Float32,1}:\n SegmentedMax(ψ = Float32[0.0, 0.0, 0.0])"

    a = Aggregation(
                    SegmentedMean(2),
                    SegmentedMax(2),
                    Aggregation(SegmentedPNorm(zeros(2) |> f32, ones(2) |> f32, -ones(2) |> f32)),
                    Aggregation(SegmentedLSE(zeros(2) |> f32, ones(2) |> f32))
                   )
    @test repr(a) == "⟨SegmentedMean(2), SegmentedMax(2), SegmentedPNorm(2), SegmentedLSE(2)⟩"
    @test repr(a; context=:compact => true) == "Aggregation(2, 2, 2, 2)"
    @test repr(MIME("text/plain"), a) == "Aggregation{Float32,4}:\n" *
        " SegmentedMean(ψ = Float32[0.0, 0.0])\n" *
        " SegmentedMax(ψ = Float32[0.0, 0.0])\n" *
        " SegmentedPNorm(ρ = Float32[0.0, 0.0], c = Float32[1.0, 1.0], ψ = Float32[-1.0, -1.0])\n" *
        " SegmentedLSE(ρ = Float32[0.0, 0.0], ψ = Float32[1.0, 1.0])"

    a = SegmentedMean(zeros(2))
    @test repr(a) == "SegmentedMean(ψ = [0.0, 0.0])"
    @test repr(a; context=:compact => true) == "SegmentedMean(2)"
    @test repr(a) == repr(MIME("text/plain"), a)
end

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

@testset "imputing dense io" begin
    X = ColImputingMatrix(reshape(1:6, 2, 3) |> Matrix)
    Y = RowImputingMatrix(reshape(1:8, 4, 2) |> Matrix)

    m = RowImputingDense(1, 1)
    @test repr(m) == "RowImputingDense(1, 1)"
    @test repr(m) == repr(m; context=:compact => true) == repr(MIME("text/plain"), m)

    m = ColImputingDense(1, 1)
    @test repr(m) == "ColImputingDense(1, 1)"
    @test repr(m) == repr(m; context=:compact => true) == repr(MIME("text/plain"), m)
end

@testset "ngram matrix io" begin
    X = NGramMatrix(["a", "b", "c"])
    @test repr(X) == "NGramMatrix(s = [\"a\", \"b\", \"c\"], n = 3, b = 256, m = 2053)"
    @test repr(X; context=:compact => true) == "2053x3 NGramMatrix"
    @test repr(MIME("text/plain"), X) == "2053×3 NGramMatrix{String}:\n \"a\"\n \"b\"\n \"c\""

    X = NGramMatrix([missing, "b", missing])
    @test repr(X) == "NGramMatrix(s = [\"a\", \"b\", \"c\"], n = 3, b = 256, m = 2053)"
    @test repr(X; context=:compact => true) == "2053x3 NGramMatrix"
    @test repr(MIME("text/plain"), X) == "2053×3 NGramMatrix{String}:\n \"a\"\n \"b\"\n \"c\""
end
