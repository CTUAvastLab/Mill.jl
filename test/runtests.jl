using Test, Mill, Flux
using Random

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

gradcheck(f, xs...) =
  all(isapprox.(ngradient(f, xs...),
                gradient(f, xs...), rtol = 1e-5, atol = 1e-5))

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)

function mngradient(f, xs::AbstractArray...)
    grads = zero.(xs)
    for (x, Δ) in zip(xs, grads), i in eachindex(x)
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

function mgradcheck(f, xs...)
    correct = true
    for (ng, ag) in zip(mngradient(f, xs...), gradient(f, xs...))
        if isnothing(ag)
            if any(ng .!= 0)
                correct = false
                @show ng
            end
        elseif !all(isapprox.(ng, ag, rtol = 1e-5, atol = 1e-5))
            correct = false
            grad_dif = [abs.(x) for x in (ng .- ag)]
            @show ng
            @show ag
            @show grad_dif
        end
    end
    correct
end

mgradtest(f, xs::AbstractArray...) = mgradcheck((xs...) -> sum(sin.(f(xs...))), xs...)

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

include("datanode.jl")
include("modelnode.jl")
include("missing.jl")
include("aggregation.jl")
include("gradtests.jl")
include("conv.jl")
include("bags.jl")
include("ngrams.jl")
include("activations.jl")
include("hierarchical_utils.jl")
include("replacein.jl")
include("partialeval.jl")
