abstract type AggregationFunction end

struct Aggregation{N} <: AggregationFunction
    fs::NTuple{N, AggregationFunction}
    Aggregation(fs::Vararg{AggregationFunction, N}) where N = new{N}(fs)
    Aggregation(fs::NTuple{N, AggregationFunction}) where N = new{N}(fs)
end

Flux.@functor Aggregation

function (a::Aggregation)(x::Union{AbstractArray, Missing}, bags::AbstractBags, args...)
    vcat(vcat([f(x, bags, args...) for f in a.fs]...), Zygote.@ignore length.(bags)')
end
(a::AggregationFunction)(x::ArrayNode, args...) = mapdata(x -> a(x, args...), x)

Base.getindex(a::Aggregation, i) = a.fs[i]

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

# more stable definitions for r_map and p_map
Zygote.@adjoint softplus(x) = softplus.(x), Δ -> (Δ .* σ.(x),)

include("segmented_sum.jl")
include("segmented_mean.jl")
include("segmented_max.jl")
include("segmented_pnorm.jl")
include("segmented_lse.jl")
# include("transformer.jl")

Base.show(io::IO, ::MIME"text/plain", a::T) where T <: AggregationFunction = print(io, "$(T.name)($(length(a.C)))")
Base.show(io::IO, m::MIME"text/plain", a::Aggregation{N}) where N = print(io, "⟨" * join(repr(m, f) for f in a.fs ", ") * "⟩")

const names = ["Sum", "Mean", "Max", "PNorm", "LSE"]
for p in powerset(collect(1:length(names)))
    s = Symbol("Segmented", names[p]...)
    @eval function $s(d::Int)
        Aggregation($((Expr(:call, Symbol("_Segmented" * n), :d)
                       for n in names[p])...))
    end
    if length(p) > 1
        @eval function $s(D::Vararg{Int, $(length(p))})
            Aggregation($((Expr(:call, Symbol("_Segmented" * n), :(D[$i]))
                           for (i,n) in enumerate(names[p]))...))
        end
    end
    @eval export $s
end
