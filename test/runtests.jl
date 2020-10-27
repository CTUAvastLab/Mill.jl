using Test
using Mill
using Mill: nobs, reflectinmodel, sparsify, mapdata
using Mill: BagConv, convsum, bagconv, legacy_bagconv, _convshift, ∇wbagconv, ∇xbagconv, ∇convsum
using Mill: ngrams, string2ngrams, countngrams, multrans, catobs
using Mill: p_map, inv_p_map, r_map, inv_r_map, bagnorm
using Base.Iterators: partition, product
using Flux
using Random
using Combinatorics
using SparseArrays
using DataFrames
using HierarchicalUtils
using BenchmarkTools: @btime

function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

# TODO rewrite this check once all ChainRule types are available
# https://github.com/FluxML/Zygote.jl/issues/603
gradcheck((ag, ng)) = isnothing(ag) ? all(ng .== 0) : isapprox(ng, ag, rtol = 1e-5, atol = 1e-5)
gradcheck(f::Function, xs...) = all(gradcheck, zip(gradient(f, xs...), ngradient(f, xs...)))

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)

Random.seed!(24)

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
        ScatteredBags([collect(7:10), [], collect(1:3), [], collect(4:6), []]),
        ScatteredBags([[], collect(1:10), []]),
]

const BAGS3 = [
         (AlignedBags([1:2, 3:4, 0:-1]), ScatteredBags([[2,3,4], [1], []]), AlignedBags([1:4, 0:-1, 5:8, 0:-1])),
         (AlignedBags([0:-1, 1:2, 3:4]), ScatteredBags([[1], [2], [3, 4]]), AlignedBags([0:-1, 1:7, 0:-1, 8:8])),
         (AlignedBags([0:-1, 0:-1, 1:2, 3:4]), ScatteredBags([[2,4], [], [3, 1], []]), AlignedBags([1:1, 2:2, 0:-1, 3:8])),
         (AlignedBags([0:-1, 1:2, 3:4, 0:-1]), ScatteredBags([[], [1,3], [2,4], []]), AlignedBags([0:-1, 1:2, 3:6, 7:8]))
        ]

for test_f in readdir(".")
    @info "<HEARTBEAT>"
    (endswith(test_f, ".jl") && test_f != "runtests.jl") || continue
    @eval @testset $test_f begin
        include($test_f)
    end
end
