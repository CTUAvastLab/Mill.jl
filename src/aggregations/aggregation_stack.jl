"""
    AggregationStack{T <: Tuple{Vararg{AbstractAggregation}}} <: AbstractAggregation

A container that implements a concatenation of one or more `AbstractAggregation`s.

Construct with e.g. `AggregationStack(SegmentedMean([t::Type, ]d))` for single
operators and with e.g. `SegmentedPNormLSE([t::Type, ]d)` for concatenations. With these calls
all parameters inside operators are initialized randomly as `Float32` arrays, unless type `t` is
further specified. Another option is to `vcat` two operators together.

Nested stacks are flattened into a single-level structure upon construction, see examples.

Intended to be used as a functor:

    (a::AggregationStack)(x, bags[, w])

where `x` is either `AbstractMatrix` or `missing`, `bags` is [`AbstractBags`](@ref) structure
and optionally `w` is an `AbstractVector` of weights.

# Examples
```jldoctest
julia> a = AggregationStack(SegmentedMean(2), SegmentedMax(2))
AggregationStack:
 SegmentedMean(ψ = Float32[0.0, 0.0])
 SegmentedMax(ψ = Float32[0.0, 0.0])

julia> a(Float32[0 1 2; 3 4 5], bags([1:1, 2:3]))
4×2 Matrix{Float32}:
 0.0  1.5
 3.0  4.5
 0.0  2.0
 3.0  5.0

julia> a = AggregationStack(SegmentedMean(2), AggregationStack(SegmentedMax(2)))
AggregationStack:
 SegmentedMean(ψ = Float32[0.0, 0.0])
 SegmentedMax(ψ = Float32[0.0, 0.0])

julia> a = SegmentedMeanMax(Float32, 2)
AggregationStack:
 SegmentedMean(ψ = Float32[0.0, 0.0])
 SegmentedMax(ψ = Float32[0.0, 0.0])

julia> vcat(SegmentedMean(2), SegmentedMax(2))
AggregationStack:
 SegmentedMean(ψ = Float32[0.0, 0.0])
 SegmentedMax(ψ = Float32[0.0, 0.0])

```

See also: [`AbstractAggregation`](@ref), [`SegmentedSum`](@ref), [`SegmentedMax`](@ref),
    [`SegmentedMean`](@ref), [`SegmentedPNorm`](@ref), [`SegmentedLSE`](@ref).
"""
struct AggregationStack{T <: Tuple{Vararg{AbstractAggregation}}} <: AbstractAggregation
    fs::T
    function AggregationStack(fs::Tuple{Vararg{AbstractAggregation}})
        ffs = _flatten_agg(fs)
        new{typeof(ffs)}(ffs)
    end
end

_flatten_agg(t) = tuple(vcat(map(_flatten_agg, t)...)...)
_flatten_agg(a::AggregationStack) = vcat(map(_flatten_agg, a.fs)...)
_flatten_agg(a::AbstractAggregation) = [a]

AggregationStack(fs::AbstractAggregation...) = AggregationStack(fs)

Flux.@layer :ignore AggregationStack

# function (a::AggregationStack)(x::Maybe{AbstractArray}, bags::AbstractBags, args...)
#     reduce(vcat, (f(x, bags, args...) for f in a.fs))
# end

@generated function (a::AggregationStack{T})(x::Maybe{AbstractArray}, bags::AbstractBags, args...) where {T<:Tuple}
    l = T.parameters |> length
    chs = map(1:l) do i
        :(a.fs[$i](x, bags, args...))
    end
    quote
        vcat($(chs...))
    end
end

Flux.@forward AggregationStack.fs Base.getindex, Base.firstindex, Base.lastindex, Base.first,
    Base.last, Base.iterate, Base.eltype

Base.length(a::AggregationStack) = sum(length.(a.fs))
Base.size(a::AggregationStack) = tuple(sum(only, size.(a.fs)))

function Base.show(io::IO, ::MIME"text/plain", a::AggregationStack)
    print(io, "AggregationStack:\n")
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
