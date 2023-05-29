using Test

using Mill
using Mill: numobs, mapdata
using Mill: BagConv, convsum, bagconv, legacy_bagconv, _convshift, ∇wbagconv, ∇xbagconv, ∇convsum
using Mill: ngrams, countngrams
using Mill: p_map, inv_p_map, r_map, inv_r_map, _bagnorm
using Mill: Maybe
using Mill: @gradtest, @pgradtest, gradf

using Base.Iterators: partition, product
using Base: CodeUnits
using ChainRulesCore
using ChainRulesCore: NotImplementedException
using Combinatorics
using DataFrames
using Documenter
using Flux
using Flux: onehot, onehotbatch
using HierarchicalUtils
using PooledArrays
using Random
using SparseArrays

using BenchmarkTools: @btime

areequal(x) = true
areequal(x, y, zs...) = isequal(x, y) && areequal(y, zs...)

Random.seed!(0)

const CHARSET = ['0':'9';'A':'Z';'a':'z';'α':'ω']

randustring(n) = randstring(CHARSET, n)

const BAGS = [
      length2bags([1, 2, 3]),
      length2bags([4, 1, 1]),
      length2bags([2, 2, 2])
]

const BAGS2 = [
        length2bags(ones(Int, 10)),
        length2bags(2 .* ones(Int, 5)),
        length2bags([5, 5]),
        length2bags([10]),
        length2bags([3, 4, 3]),
        AlignedBags([1:3, 0:-1, 0:-1, 4:7, 0:-1, 8:10]),
        AlignedBags([0:-1, 1:5, 0:-1, 0:-1, 0:-1, 6:10]),
        ScatteredBags([collect(1:3), collect(7:10), collect(4:6)]),
        ScatteredBags([[6, 5, 4], Int[], [1, 2, 3, 1], Int[], [10, 9, 8, 7, 7], Int[]]),
        ScatteredBags([Int[], collect(1:8), Int[], fill(1, 5)]),
]

const BAGS3 = [
         (AlignedBags([1:2, 3:4, 0:-1]),
          ScatteredBags([[2, 3, 4], [1, 1], Int[]]),
          AlignedBags([1:4, 0:-1, 5:8, 0:-1])),
         (AlignedBags([0:-1, 1:2, 3:4]),
          ScatteredBags([[1], [2], [3, 4, 3]]),
          AlignedBags([0:-1, 1:7, 0:-1, 8:8])),
         (AlignedBags([0:-1, 0:-1, 1:2, 3:4]),
          ScatteredBags([[2, 4], Int[], [3, 1], Int[]]),
          AlignedBags([1:1, 2:2, 0:-1, 3:8])),
         (AlignedBags([0:-1, 1:2, 3:4, 0:-1]),
          ScatteredBags([Int[], [1, 3], [2, 2, 4], Int[]]),
          AlignedBags([0:-1, 1:2, 3:6, 7:8]))
        ]

function Mill.unpack2mill(ds::LazyNode{:Sentence})
    s = split.(ds.data, " ")
    x = NGramMatrix(reduce(vcat, s))
    BagNode(x, Mill.length2bags(length.(s)))
end

_init_agg(t::Type{<:Number}, d) = randn(t, d)
_init_agg(t::Type{<:Integer}, d) = rand(t, d)

# initialize to randn to test a lot of values even though we then initialize to zero
nonparam_aggregations(t::Type, d) = vcat(
                                 SegmentedSum(_init_agg(t, d)),
                                 SegmentedMean(_init_agg(t, d)),
                                 SegmentedMax(_init_agg(t, d)))

param_aggregations(t::Type, d) = vcat(
                                      SegmentedPNorm(_init_agg(t, d), _init_agg(t, d), _init_agg(t, d)),
                                      SegmentedLSE(_init_agg(t, d), _init_agg(t, d)))

all_aggregations(t::Type, d) = vcat(nonparam_aggregations(t, d), param_aggregations(t, d))

@testset "Doctests" begin
    DocMeta.setdocmeta!(Mill, :DocTestSetup, quote
        using Mill, Flux, Random, SparseArrays, Setfield, HierarchicalUtils
        # do not shorten prints in doctests
        ENV["LINES"] = ENV["COLUMNS"] = typemax(Int)
    end; recursive=true)
    doctest(Mill)
end

for test_f in readdir(".")
    (endswith(test_f, ".jl") && test_f ≠ "runtests.jl") || continue
    @testset verbose = true "$test_f" begin
        include(test_f)
    end
end
