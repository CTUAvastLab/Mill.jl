abstract type AggregationOperator{T <: Number} end

struct Aggregation{T, N}
    fs::NTuple{N, AggregationOperator{T}}
    Aggregation(fs::Union{Aggregation, AggregationOperator}...) = Aggregation(fs)
    function Aggregation(fs::Tuple{Vararg{Union{Aggregation{T}, AggregationOperator{T}}}}) where {T}
        ffs = _flatten_agg(fs)
        new{T, length(ffs)}(ffs)
    end
end

_flatten_agg(t) = tuple(vcat(map(_flatten_agg, t)...)...)
_flatten_agg(a::Aggregation) = vcat(map(_flatten_agg, a.fs)...)
_flatten_agg(a::AggregationOperator) = [a]

Flux.@functor Aggregation

function (a::Aggregation{T})(x::Union{AbstractArray, Missing}, bags::AbstractBags, args...) where T
    o = vcat([f(x, bags, args...) for f in a.fs]...)
    _bagcount[] ? vcat(o, Zygote.@ignore permutedims(log.(one(T) .+ length.(bags)))) : o
end
(a::Union{AggregationOperator, Aggregation})(x::ArrayNode, args...) = mapdata(x -> a(x, args...), x)

Flux.@forward Aggregation.fs Base.getindex, Base.firstindex, Base.lastindex, Base.first, Base.last, Base.iterate, Base.eltype

Base.length(a::Aggregation) = sum(length.(a.fs))
Base.size(a::Aggregation) = tuple(sum(only, size.(a.fs)))
Base.vcat(as::Aggregation...) = reduce(vcat, as |> collect)
function Base.reduce(::typeof(vcat), as::Vector{<:Aggregation})
    Aggregation(tuple(vcat((collect(a.fs) for a in as)...)...))
end

function Base.show(io::IO, @nospecialize a::Aggregation)
    if get(io, :compact, false)
        print(io, "Aggregation(", join(length.(a.fs), ", "), ")")
    else
        print(io, "⟨" * join(repr(f; context=:compact => true) for f in a.fs ", ") * "⟩")
    end
end

function Base.show(io::IO, ::MIME"text/plain", @nospecialize a::T) where T <: Aggregation
    print(io, T, ":\n")
    print_array(io, a.fs |> collect)
end

function Base.show(io::IO, @nospecialize a::T) where T <: AggregationOperator
    if get(io, :compact, false)
        print(io, nameof(T), "(", length(a), ")")
    else
        _show(io, a)
    end
end

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
# include("transformer.jl")

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
