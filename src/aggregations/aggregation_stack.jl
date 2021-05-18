"""
    AggregationStack{T, U <: Tuple{Vararg{AbstractAggregation{T}}}} <: AbstractAggregation{T}

A container that implements a concatenation of one or more `AbstractAggregation`s.

Construct with e.g. `mean_aggregation([t::Type, ]d)`, `max_aggregation([t::Type, ]d)` for single
operators and with e.g. `pnormlse_aggregation([t::Type, ]d)` for concatenations. With these calls
all parameters inside operators are initialized randomly as `Float32` arrays, unless type `t` is
further specified. It is also possible to call the constructor directly, see Examples.

Intended to be used as a functor:

    (a::AggregationStack)(x, bags[, w])

where `x` is either `Missing`, `AbstractMatrix` or [`ArrayNode`](@ref),
`bags` is [`AbstractBags`](@ref) structure and optionally `w` is an `AbstractVector` of weights.

TODO mention that it can be constructed by vcatting operators as well.
mention flattening

# Examples
```jldoctest
julia> a = AggregationStack(SegmentedMean(2), SegmentedMax(2))
AggregationStack{Float32}:
 SegmentedMean(ψ = Float32[0.0, 0.0])
 SegmentedMax(ψ = Float32[0.0, 0.0])

julia> a(Float32[0 1 2; 3 4 5], bags([1:1, 2:3]))
4×2 Matrix{Float32}:
 0.0  1.5
 3.0  4.5
 0.0  2.0
 3.0  5.0

julia> a = AggregationStack(SegmentedMean(2), AggregationStack(SegmentedMax(2)))
AggregationStack{Float32}:
 SegmentedMean(ψ = Float32[0.0, 0.0])
 SegmentedMax(ψ = Float32[0.0, 0.0])

julia> a = SegmentedMeanMax(Float32, 2)
AggregationStack{Float32}:
 SegmentedMean(ψ = Float32[0.0, 0.0])
 SegmentedMax(ψ = Float32[0.0, 0.0])

julia> vcat(SegmentedMean(2), SegmentedMax(2))
AggregationStack{Float32}:
 SegmentedMean(ψ = Float32[0.0, 0.0])
 SegmentedMax(ψ = Float32[0.0, 0.0])

```

See also: [`AbstractAggregation`](@ref), [`SegmentedSum`](@ref), [`SegmentedMax`](@ref),
    [`SegmentedMean`](@ref), [`SegmentedPNorm`](@ref), [`SegmentedLSE`](@ref).
"""
struct AggregationStack{T, U <: Tuple{Vararg{AbstractAggregation{T}}}} <: AbstractAggregation{T}
    fs::U
    function AggregationStack(fs::Tuple{Vararg{AbstractAggregation{T}}}) where T
        ffs = _flatten_agg(fs)
        new{T, typeof(ffs)}(ffs)
    end
end

_flatten_agg(t) = tuple(vcat(map(_flatten_agg, t)...)...)
_flatten_agg(a::AggregationStack) = vcat(map(_flatten_agg, a.fs)...)
_flatten_agg(a::AbstractAggregation) = [a]

AggregationStack(fs::AbstractAggregation{T}...) where T = AggregationStack(fs)

Flux.@functor AggregationStack

function (a::AggregationStack)(x::Union{AbstractArray, Missing}, bags::AbstractBags, args...)
    reduce(vcat, (f(x, bags, args...) for f in a.fs))
end

Flux.@forward AggregationStack.fs Base.getindex, Base.firstindex, Base.lastindex, Base.first,
    Base.last, Base.iterate, Base.eltype

Base.length(a::AggregationStack) = sum(length.(a.fs))
Base.size(a::AggregationStack) = tuple(sum(only, size.(a.fs)))

function Base.show(io::IO, ::MIME"text/plain", a::AggregationStack{T}) where T
    print(io, "AggregationStack{$T}:\n")
    Base.print_array(io, a.fs |> collect)
end

function Base.show(io::IO, a::AggregationStack)
    if get(io, :compact, false)
        print(io, "AggregationStack")
    else
        s = join(repr.(a.fs; context=io), "; ")
        print(io, "[", s, "]")
    end
end
