@testset "data and model node io" begin
    an = ArrayNode(ones(2,5))
    @test repr(an) == "ArrayNode(2×5 Array with Float64 elements)"
    @test repr(an; context=:compact => true) == "ArrayNode"
    @test repr(MIME("text/plain"), an) == 
        """
        2×5 ArrayNode{Matrix{Float64}, Nothing}:
         1.0  1.0  1.0  1.0  1.0
         1.0  1.0  1.0  1.0  1.0"""

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
          └── ArrayNode(2×5 Array with Float64 elements)"""

    bnm = reflectinmodel(bn, d -> Dense(d, 10), SegmentedMean)
    @test repr(bnm) == "BagModel … ↦ SegmentedMean(10) ↦ ArrayModel(Dense(10, 10))"
    @test repr(bnm; context=:compact => true) == "BagModel"
    @test repr(MIME("text/plain"), bnm) == 
        """
        BagModel … ↦ SegmentedMean(10) ↦ ArrayModel(Dense(10, 10))
          └── ArrayModel(Dense(2, 10))"""

    wbn = WeightedBagNode(an, bags([1:2, 3:5]), rand(5))
    @test repr(wbn) == "WeightedBagNode with 2 obs"
    @test repr(wbn; context=:compact => true) == "WeightedBagNode"
    @test repr(MIME("text/plain"), wbn) == 
        """
        WeightedBagNode with 2 obs
          └── ArrayNode(2×5 Array with Float64 elements)"""

    wbnm = reflectinmodel(wbn)
    @test repr(wbnm) == "BagModel … ↦ BagCount([SegmentedMean(10); SegmentedMax(10)]) ↦ ArrayModel(Dense(21, 10))"
    @test repr(wbnm; context=:compact => true) == "BagModel"
    @test repr(MIME("text/plain"), wbnm) == 
        """
        BagModel … ↦ BagCount([SegmentedMean(10); SegmentedMax(10)]) ↦ ArrayModel(Dense(21, 10))
          └── ArrayModel(Dense(2, 10))"""

    pn = ProductNode((; bn, wbn))
    @test repr(pn) == "ProductNode with 2 obs"
    @test repr(pn; context=:compact => true) == "ProductNode"
    @test repr(MIME("text/plain"), pn) ==
        """
        ProductNode with 2 obs
          ├─── bn: BagNode with 2 obs
          │          └── ArrayNode(2×5 Array with Float64 elements)
          └── wbn: WeightedBagNode with 2 obs
                     └── ArrayNode(2×5 Array with Float64 elements)"""

    pnm = reflectinmodel(pn, d -> Dense(d, 10), BagCount ∘ SegmentedMean)
    @test repr(pnm) == "ProductModel … ↦ ArrayModel(Dense(20, 10))"
    @test repr(pnm; context=:compact => true) == "ProductModel"
    @test repr(MIME("text/plain"), pnm) ==
        """
        ProductModel … ↦ ArrayModel(Dense(20, 10))
          ├─── bn: BagModel … ↦ BagCount(SegmentedMean(10)) ↦ ArrayModel(Dense(11, 10))
          │          └── ArrayModel(Dense(2, 10))
          └── wbn: BagModel … ↦ BagCount(SegmentedMean(10)) ↦ ArrayModel(Dense(11, 10))
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
          └── BagModel … ↦ BagCount([SegmentedMean(10); SegmentedMax(10)]) ↦ ArrayModel(Dense(21, 10))
                └── ArrayModel(Dense(2053, 10))"""
end

@testset "aggregation io" begin
    a = SegmentedMax(3)
    @test repr(a) == "SegmentedMax(3)"
    @test repr(a; context=:compact => true) == "SegmentedMax"
    @test repr(MIME("text/plain"), a) == "SegmentedMax(ψ = Float32[0.0, 0.0, 0.0])"

    a = AggregationStack(
                    SegmentedMean(2),
                    SegmentedMax(2),
                    AggregationStack(SegmentedPNorm(zeros(2) |> f32, ones(2) |> f32, -ones(2) |> f32)),
                    AggregationStack(SegmentedLSE(zeros(2) |> f32, ones(2) |> f32))
                   )
    @test repr(a) == "[SegmentedMean(2); SegmentedMax(2); SegmentedPNorm(2); SegmentedLSE(2)]"
    @test repr(a; context=:compact => true) == "AggregationStack"
    @test repr(MIME("text/plain"), a) == 
        """
        AggregationStack:
         SegmentedMean(ψ = Float32[0.0, 0.0])
         SegmentedMax(ψ = Float32[0.0, 0.0])
         SegmentedPNorm(ψ = Float32[0.0, 0.0], ρ = Float32[1.0, 1.0], c = Float32[-1.0, -1.0])
         SegmentedLSE(ψ = Float32[0.0, 0.0], ρ = Float32[1.0, 1.0])"""

    bc = BagCount(SegmentedMean(3))
    @test repr(bc) == "BagCount(SegmentedMean(3))"
    @test repr(bc; context=:compact => true) == "BagCount"
    @test repr(MIME("text/plain"), bc) == "BagCount(SegmentedMean(ψ = Float32[0.0, 0.0, 0.0]))"

    bc = BagCount(SegmentedMeanMax(3))
    @test repr(bc) == "BagCount([SegmentedMean(3); SegmentedMax(3)])"
    @test repr(bc; context=:compact => true) == "BagCount"
    @test repr(MIME("text/plain"), bc) ==
        """
        BagCount(AggregationStack:
         SegmentedMean(ψ = Float32[0.0, 0.0, 0.0])
         SegmentedMax(ψ = Float32[0.0, 0.0, 0.0])
        )"""
end

@testset "maybe hot vector io" begin
    x = MaybeHotVector(1, 3)
    @test repr(x) == "MaybeHotVector(i = 1, l = 3)"
    @test repr(x; context=:compact => true) == "3-element MaybeHotVector"
    @test repr(MIME("text/plain"), x) ==
        """
        3-element MaybeHotVector{Int64, Int64, Bool}:
         1
         0
         0"""

    x = MaybeHotVector(missing, 4)
    @test repr(x) == "MaybeHotVector(i = missing, l = 4)"
    @test repr(x; context=:compact => true) == "4-element MaybeHotVector"
    @test repr(MIME("text/plain"), x) ==
        """
        4-element MaybeHotVector{Missing, Int64, Missing}:
         missing
         missing
         missing
         missing"""
end

@testset "maybe hot matrix io" begin
    X = MaybeHotMatrix([1, 3], 3)
    @test repr(X) == "MaybeHotMatrix(I = [1, 3], l = 3)"
    @test repr(X; context=:compact => true) == "3×2 MaybeHotMatrix"
    @test repr(MIME("text/plain"), X) ==
        """
        3×2 MaybeHotMatrix{Int64, Int64, Bool}:
         1  0
         0  0
         0  1"""

    X = MaybeHotMatrix([missing, 2], 2)
    @test repr(X) == "MaybeHotMatrix(I = Union{Missing, Int64}[missing, 2], l = 2)"
    @test repr(X; context=:compact => true) == "2×2 MaybeHotMatrix"
    @test repr(MIME("text/plain"), X) ==
        """
        2×2 MaybeHotMatrix{Union{Missing, Int64}, Int64, Union{Missing, Bool}}:
         missing  false
         missing   true"""

    X = MaybeHotMatrix([missing, missing, missing], 4)
    @test repr(X) == "MaybeHotMatrix(I = [missing, missing, missing], l = 4)"
    @test repr(X; context=:compact => true) == "4×3 MaybeHotMatrix"
    @test repr(MIME("text/plain"), X) ==
        """
        4×3 MaybeHotMatrix{Missing, Int64, Missing}:
         missing  missing  missing
         missing  missing  missing
         missing  missing  missing
         missing  missing  missing"""
end

@testset "pre imputing matrix io" begin
    X = PreImputingMatrix(reshape(1:8, 4, 2) |> Matrix)
    @test repr(X) == "PreImputingMatrix(W = [1 5; 2 6; 3 7; 4 8], ψ = [0, 0])"
    @test repr(X; context=:compact => true) == "4×2 PreImputingMatrix"
    @test repr(MIME("text/plain"), X) == 
        """
        4×2 PreImputingMatrix{Int64, Matrix{Int64}, Vector{Int64}}:
        W:
         1  5
         2  6
         3  7
         4  8

        ψ:
         0  0"""
end

@testset "post imputing matrix io" begin
    X = PostImputingMatrix(reshape(1:6, 2, 3) |> Matrix)
    @test repr(X) == "PostImputingMatrix(W = [1 3 5; 2 4 6], ψ = [0, 0])"
    @test repr(X; context=:compact => true) == "2×3 PostImputingMatrix"
    @test repr(MIME("text/plain"), X) == 
        """
        2×3 PostImputingMatrix{Int64, Matrix{Int64}, Vector{Int64}}:
        W:
         1  3  5
         2  4  6

        ψ:
         0
         0"""
end

@testset "imputing dense io" begin
    X = PostImputingMatrix(reshape(1:6, 2, 3) |> Matrix)
    Y = PreImputingMatrix(reshape(1:8, 4, 2) |> Matrix)

    m = preimputing_dense(1, 1)
    @test repr(m) == "[pre_imputing]Dense(1, 1)"
    @test repr(m) == repr(m; context=:compact => true) == repr(MIME("text/plain"), m)

    m = postimputing_dense(1, 1)
    @test repr(m) == "[post_imputing]Dense(1, 1)"
    @test repr(m) == repr(m; context=:compact => true) == repr(MIME("text/plain"), m)
end

@testset "ngram matrix io" begin
    X = NGramMatrix(["a", "b", "c"])
    @test repr(X) == "NGramMatrix(S = [\"a\", \"b\", \"c\"], n = 3, b = 256, m = 2053)"
    @test repr(X; context=:compact => true) == "2053×3 NGramMatrix"
    @test repr(MIME("text/plain"), X) == 
        """
        2053×3 NGramMatrix{String, Vector{String}, Int64}:
         \"a\"
         \"b\"
         \"c\""""

    X = NGramMatrix([missing, "b", missing] |> PooledArray)
    @test repr(X) == "NGramMatrix(S = Union{Missing, String}[missing, \"b\", missing], n = 3, b = 256, m = 2053)"
    @test repr(X; context=:compact => true) == "2053×3 NGramMatrix"
    @test repr(MIME("text/plain"), X) ==
        """
        2053×3 NGramMatrix{Union{Missing, String}, PooledVector{Union{Missing, String}, UInt32, Vector{UInt32}}, Union{Missing, Int64}}:
         missing
         \"b\"
         missing"""

    X = NGramMatrix([codeunits("hello"), codeunits("world")])
    @test repr(X) == "NGramMatrix(S = CodeUnits{UInt8, String}" *
        "[[0x68, 0x65, 0x6c, 0x6c, 0x6f], [0x77, 0x6f, 0x72, 0x6c, 0x64]], n = 3, b = 256, m = 2053)"
    @test repr(X; context=:compact => true) == "2053×2 NGramMatrix"
    @test repr(MIME("text/plain"), X) ==
        """
        2053×2 NGramMatrix{CodeUnits{UInt8, String}, Vector{CodeUnits{UInt8, String}}, Int64}:
         UInt8[0x68, 0x65, 0x6c, 0x6c, 0x6f]
         UInt8[0x77, 0x6f, 0x72, 0x6c, 0x64]"""

    X = NGramMatrix([[1,2,3], missing, missing])
    @test repr(X) == "NGramMatrix(S = Union{Missing, Vector{Int64}}[[1, 2, 3], missing, missing], n = 3, b = 256, m = 2053)"
    @test repr(X; context=:compact => true) == "2053×3 NGramMatrix"
    @test repr(MIME("text/plain"), X) ==
        """
        2053×3 NGramMatrix{Union{Missing, Vector{Int64}}, Vector{Union{Missing, Vector{Int64}}}, Union{Missing, Int64}}:
         [1, 2, 3]
         missing
         missing"""
end
