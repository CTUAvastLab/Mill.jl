abstract type AggregationFunction end

struct Aggregation{N} <: AggregationFunction
    fs::NTuple{N, AggregationFunction}
    Aggregation(fs::Vararg{AggregationFunction, N}) where N = new{N}(fs)
    Aggregation(fs::NTuple{N, AggregationFunction}) where N = new{N}(fs)
end

Flux.@functor Aggregation

function (a::Aggregation)(x::Union{AbstractArray, Missing}, bags::AbstractBags, args...)
    o = vcat([f(x, bags, args...) for f in a.fs]...)
    _bagcount[] ? vcat(o, Zygote.@ignore length.(bags)') : o
end
(a::AggregationFunction)(x::ArrayNode, args...) = mapdata(x -> a(x, args...), x)

Base.getindex(a::Aggregation, i) = a.fs[i]

@inline bagnorm(w::Nothing, b) = length(b)
@inline bagnorm(w::AbstractVector, b) = @views sum(w[b])
@inline bagnorm(w::AbstractMatrix, b) = @views vec(sum(w[:, b], dims=2))

@inline weight(w::Nothing, _, _, ::Type{T}) where T = one(T)
@inline weight(w::AbstractVector, _, j, _) = w[j]
@inline weight(w::AbstractMatrix, i, j, _) = w[i, j]

@inline weightsum(ws::Real, _) = ws
@inline weightsum(ws::AbstractVector, i) = ws[i]

# more stable definitions for r_map and p_map
rrule(::typeof(softplus), x) = softplus.(x), Δ -> (NO_FIELDS, Δ .* σ.(x))

include("segmented_sum.jl")
include("segmented_mean.jl")
include("segmented_max.jl")
include("segmented_pnorm.jl")
include("segmented_lse.jl")

Base.show(io::IO, ::MIME"text/plain", a::T) where T <: AggregationFunction = print(io, "$(T.name)($(length(a.ψ)))")
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
