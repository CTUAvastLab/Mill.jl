import Base: show, getindex

abstract type AggregationFunction end

struct Aggregation{N} <: AggregationFunction
    fs::NTuple{N, AggregationFunction}
    Aggregation(fs::Vararg{AggregationFunction, N}) where N = new{N}(fs)
    Aggregation(fs::NTuple{N, AggregationFunction}) where N = new{N}(fs)
end

Flux.@functor Aggregation

(a::Aggregation)(args...) = vcat([f(args...) for f in a.fs]...)

(a::AggregationFunction)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)

Base.show(io::IO, a::AggregationFunction) = modelprint(io, a)
Base.getindex(a::AggregationFunction, i) = a.fs[i]

function modelprint(io::IO, a::Aggregation{N}; pad=[]) where N
    paddedprint(io, N == 1 ? "" : "⟨")
    for f in a.fs[1:end-1]
        modelprint(io, f, pad=pad)
        paddedprint(io, ", ")
    end
    modelprint(io, a.fs[end], pad=pad)
    paddedprint(io, (N == 1 ? "" : "⟩"))
end

const AggregationWeights = Union{Nothing,
                                 AbstractVector{T} where T <: Real,
                                 AbstractMatrix{T} where T <: Real}
const MaybeMatrix = Union{Missing,
                          AbstractMatrix{T} where T <: Real}

bagnorm(w::Nothing, b) = length(b)
bagnorm(w::AbstractVector, b) = @views sum(w[b])
bagnorm(w::AbstractMatrix, b) = @views vec(sum(w[:, b], dims=2))

weight(w::Nothing, _, _) = 1
weight(w::AbstractVector, _, j) = w[j]
weight(w::AbstractMatrix, i, j) = w[i, j]

weightsum(ws::Real, _) = ws
weightsum(ws::AbstractVector, i) = ws[i]

include("segmented_sum.jl")
include("segmented_mean.jl")
include("segmented_max.jl")
include("segmented_pnorm.jl")
include("segmented_lse.jl")

export SegmentedSum, SegmentedMean, SegmentedMax, SegmentedPNorm, SegmentedLSE

const names = ["Sum", "Mean", "Max", "PNorm", "LSE"]
for idxs in powerset(collect(1:length(names)))
    length(idxs) > 1 || continue
    for p in permutations(idxs)
        s = Symbol("Segmented", names[p]...)
        @eval function $s(d::Int)
            Aggregation($((Expr(:call, Symbol("Segmented" * n), :d)
                           for n in names[p])...))
        end
        @eval function $s(D::Vararg{Int, $(length(p))})
            Aggregation($((Expr(:call, Symbol("Segmented" * n), :(D[$i]))
                           for (i,n) in enumerate(names[p]))...))
        end
        @eval export $s
    end
end
