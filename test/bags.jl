@testset "Constructors" begin
    @test AlignedBags().bags == UnitRange{Int}[]
    @test AlignedBags([1]).bags == [1:1]
    @test AlignedBags([1, 1]).bags == [1:2]
    @test AlignedBags([1, 1, 2, 2, 2, 3]).bags == [1:2, 3:5, 6:6]
    @test AlignedBags([3, 3, 1, 1, 1, 2]).bags == [1:2, 3:5, 6:6]
    @test AlignedBags([2, 2, 3, 3, 3, 1]).bags == [1:2, 3:5, 6:6]
    @test AlignedBags(UInt64.([2, 2, 3, 3, 3, 1])).bags == [1:2, 3:5, 6:6]
    @test_throws ArgumentError AlignedBags([1, 1, 1, 2, 2, 1])
    @test_throws ArgumentError AlignedBags([2, 1, 1, 1, 1, 2])

    @test ScatteredBags().bags == Int[]
    @test ScatteredBags([1]).bags == [[1]]
    @test ScatteredBags([1, 2, 1]).bags == [[1, 3], [2]]
    @test ScatteredBags([1, 2, 3, 1, 2, 3]).bags == [[1,4], [2,5], [3,6]]
    @test ScatteredBags([1, 1, 2, 2, 2, 3]).bags == [[1,2], [3,4,5], [6]]
    @test ScatteredBags([3, 3, 1, 1, 1, 2]).bags == [[3,4,5], [6], [1,2]]
    @test ScatteredBags(UInt64.([3, 3, 1, 1, 1, 2])).bags == [[3,4,5], [6], [1,2]]
    @test ScatteredBags([2, 2, 3, 3, 3, 1]).bags == [[6], [1,2], [3,4,5]]
end

@testset "bags()" begin
    @test Mill.bags(Int[]) isa AlignedBags
    @test Mill.bags([1, 1, 1, 2, 2, 3, 3]) isa AlignedBags
    @test Mill.bags([2, 2, 2, 1, 1, 3, 3]) isa AlignedBags
    @test Mill.bags([1, 2, 1]) isa ScatteredBags
end

@testset "length2bags" begin
    @test length2bags([1, 3, 2]).bags == [1:1,2:4,5:6]
    @test length2bags([1, 3, 0, 2]).bags == [1:1,2:4,0:-1,5:6]
    @test length2bags([2]).bags == [1:2]
    @test length2bags([1]).bags == [1:1]
    @test length2bags([0]).bags == [0:-1]
end

@testset "aligned hcat" begin
    b1 = AlignedBags([1:1, 2:5])
    b2 = AlignedBags([1:2, 3:3])
    b3 = AlignedBags([0:-1, 1:3, 0:-1, 4:6])
    b4 = AlignedBags(Int[])

    @test vcat(b1, b1).bags == [1:1, 2:5, 6:6, 7:10]
    @test vcat(b1, b2).bags == [1:1, 2:5, 6:7, 8:8]
    @test vcat(b1, b3).bags == [1:1, 2:5, 0:-1, 6:8, 0:-1, 9:11]
    @test vcat(b1, b4).bags == [1:1, 2:5]
    @test vcat(b2, b1).bags == [1:2, 3:3, 4:4, 5:8]
    @test vcat(b2, b2).bags == [1:2, 3:3, 4:5, 6:6]
    @test vcat(b2, b3).bags == [1:2, 3:3, 0:-1, 4:6, 0:-1, 7:9]
    @test vcat(b2, b4).bags == [1:2, 3:3]
    @test vcat(b3, b1).bags == [0:-1, 1:3, 0:-1, 4:6, 7:7, 8:11]
    @test vcat(b3, b2).bags == [0:-1, 1:3, 0:-1, 4:6, 7:8, 9:9]
    @test vcat(b3, b3).bags == [0:-1, 1:3, 0:-1, 4:6, 0:-1, 7:9, 0:-1, 10:12]
    @test vcat(b3, b4).bags == [0:-1, 1:3, 0:-1, 4:6]
    @test vcat(b4, b1).bags == b1.bags
    @test vcat(b4, b2).bags == b2.bags
    @test vcat(b4, b3).bags == b3.bags
    @test vcat(b4, b4).bags == b4.bags
    @test vcat(b1, b2, b3, b4).bags == vcat(vcat(b1, b2), vcat(b3, b4)).bags
end

@testset "scattered hcat" begin
    b1 = ScatteredBags([[1], [1, 3, 2], Int[]])
    b2 = ScatteredBags([Int[], [3, 2, 1], [4, 1, 2]])
    b3 = ScatteredBags([[1], [2], [3]])
    b4 = ScatteredBags(Int[])

    @test vcat(b1, b1).bags == [[1], [1,3,2], Int[], [4], [4,6,5], Int[]]
    @test vcat(b1, b2).bags == [[1], [1,3,2], Int[], Int[], [6,5,4], [7,4,5]]
    @test vcat(b1, b3).bags == [[1], [1,3,2], Int[], [4], [5], [6]]
    @test vcat(b1, b4).bags == [[1], [1,3,2], Int[]]
    @test vcat(b2, b1).bags == [Int[], [3,2,1], [4,1,2], [5], [5,7,6], Int[]]
    @test vcat(b2, b2).bags == [Int[], [3,2,1], [4,1,2], Int[], [7,6,5], [8,5,6]]
    @test vcat(b2, b3).bags == [Int[], [3,2,1], [4,1,2], [5], [6], [7]]
    @test vcat(b2, b4).bags == [Int[], [3,2,1], [4,1,2]]
    @test vcat(b3, b1).bags == [[1], [2], [3], [4], [4,6,5], Int[]]
    @test vcat(b3, b2).bags == [[1], [2], [3], Int[], [6,5,4], [7,4,5]]
    @test vcat(b3, b3).bags == [[1], [2], [3], [4], [5], [6]]
    @test vcat(b3, b4).bags == [[1], [2], [3]]
    @test vcat(b4, b1).bags == b1.bags
    @test vcat(b4, b2).bags == b2.bags
    @test vcat(b4, b3).bags == b3.bags
    @test vcat(b4, b4).bags == b4.bags
    @test vcat(b1, b2, b3, b4).bags == vcat(vcat(b1, b2), vcat(b3, b4)).bags
end

@testset "testing remapping for aligned bags" begin
    b = AlignedBags([1:1,2:3,4:5])
    @test remapbags(b, [2,3])[1].bags == [1:2,3:4]
    @test remapbags(b, 2:3)[1].bags == [1:2,3:4]
    @test remapbags(b, [2,3])[2] == [2,3,4,5]
    @test remapbags(b, 2:3)[2] == [2,3,4,5]

    b = AlignedBags([1:2,3:3,4:5])
    @test remapbags(b, [1,3])[1].bags == [1:2,3:4]
    @test remapbags(b, [1,3])[2] == [1,2,4,5]
    @test remapbags(b, [2,3])[1].bags == [1:1,2:3]
    @test remapbags(b, [2,3])[2] == [3,4,5]

    b = AlignedBags([1:2,0:-1,3:4])
    @test remapbags(b, [2,3])[1].bags == [0:-1,1:2]
    @test remapbags(b, [2,3])[2] == [3,4]
end

@testset "testing remapping for scattered bags" begin
    b = ScatteredBags([[1], Int[], [1,5], [4,3,5]])
    @test remapbags(b, [1])[1].bags == remapbags(b, 1:1)[1].bags
    @test remapbags(b, [1])[2] == remapbags(b, 1:1)[2]
    @test remapbags(b, [1, 2, 3])[1].bags == remapbags(b, 1:3)[1].bags
    @test remapbags(b, [1, 2, 3])[2] == remapbags(b, 1:3)[2]

    @test remapbags(b, [1,2,3,4])[1].bags == [[1], Int[], [1,2], [3,4,2]]
    @test remapbags(b, [1,2,3,4])[2] == [1,5,4,3]
    @test remapbags(b, [1,1])[1].bags == [[1], [1]]
    @test remapbags(b, [1,1])[2] == [1]
    @test remapbags(b, [1,2])[1].bags == [[1], Int[]]
    @test remapbags(b, [1,2])[2] == [1]
    @test remapbags(b, [1,3])[1].bags == [[1], [1,2]]
    @test remapbags(b, [1,3])[2] == [1,5]
    @test remapbags(b, [1,4])[1].bags == [[1], [2,3,4]]
    @test remapbags(b, [1,4])[2] == [1,4,3,5]
    @test remapbags(b, [2,1])[1].bags == [Int[], [1]]
    @test remapbags(b, [2,1])[2] == [1]
    @test remapbags(b, [2,2])[1].bags == [Int[], Int[]]
    @test remapbags(b, [2,2])[2] == Int[]
    @test remapbags(b, [2,3])[1].bags == [Int[], [1,2]]
    @test remapbags(b, [2,3])[2] == [1,5]
    @test remapbags(b, [2,4])[1].bags == [Int[], [1,2,3]]
    @test remapbags(b, [2,4])[2] == [4,3,5]
end

@testset "equals and hash" begin
    a = ScatteredBags([[1], Int[], [1,4], [4,3,5]])
    b = ScatteredBags([[1], Int[], [1,5], [4,3,5]])
    c = ScatteredBags([[1], Int[], [1,5], [4,3,5]])
    @test a != b
    @test a != c
    @test b == c
    @test hash(a) !== hash(b)
    @test hash(b) === hash(c)
end

@testset "length." begin
    @test length.(AlignedBags(Int[])) == Int[]
    @test length.(AlignedBags([1:2, 0:-1, 3:5])) == [2, 0, 3]
    @test length.(ScatteredBags(Int[])) == Int[]
    @test length.(ScatteredBags([[1,2], Int[], [3,4,5]])) == [2, 0, 3]
end

@testset "length type stability" begin
    @test length.(AlignedBags(Int[])) isa Vector{Int}
    @test length.(AlignedBags([1:2, 0:-1, 3:5])) isa Vector{Int}
    @test length.(ScatteredBags(Int[])) isa Vector{Int}
    @test length.(ScatteredBags([[1,2], Int[], [3,4,5]])) isa Vector{Int}
end

