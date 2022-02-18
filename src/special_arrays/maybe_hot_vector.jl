"""
    MaybeHotVector{T, U, V} <: AbstractVector{V}

A vector-like structure for representing one-hot encoded variables.
Like `Flux.OneHotVector` but supports `missing` values.

Construct with the [`maybehot`](@ref) function.

See also: [`MaybeHotMatrix`](@ref), [`maybehotbatch`](@ref).
"""
struct MaybeHotVector{T, U, V} <: AbstractVector{V}
    i::T
    l::U

    MaybeHotVector(i::T, l::U) where {T <: Integer, U <: Integer} = new{T, U, Bool}(i, l)
    MaybeHotVector(i::T, l::U) where {T <: Missing, U <: Integer} = new{T, U, Missing}(i, l)
    MaybeHotVector(i::T, l::U) where {T <: Maybe{Integer}, U <: Integer} = new{T, U, Maybe{T}}(i, l)

    function MaybeHotVector{T, U, V}(i, l) where {T <: Maybe{Integer}, U <: Integer, V <: Maybe{Bool}}
        new{T, U, V}(convert(T, i), convert(U, l))
    end
end

Base.size(x::MaybeHotVector) = (x.l,)
Base.length(x::MaybeHotVector) = x.l
Base.getindex(x::MaybeHotVector, i::Integer) = (@boundscheck checkbounds(x, i); x.i == i)
Base.getindex(x::MaybeHotVector, ::Colon) = MaybeHotVector(x.i, x.l)

function Base.convert(::Type{MaybeHotVector{T, U, W}}, x::MaybeHotVector) where {T, U, W}
    MaybeHotVector{T, U, W}(convert(T, x.i), convert(U, x.l))
end

function Base.promote_rule(::Type{MaybeHotVector{T, U, V}},
        ::Type{MaybeHotVector{A, B, C}}) where {T, A, U, B, V, C}
    MaybeHotVector{promote_type(T, A), promote_type(U, B), promote_type(V, C)}
end

function _check_l(xs::AbstractVecOrTuple{MaybeHotVector})
    l = xs[1].l
    if any(!isequal(l), (x.l for x in xs))
        DimensionMismatch(
            "Number of rows of `MaybeHotMatrix`s to hcat must correspond"
        ) |> throw
    end
    l
end

Base.hcat(xs::T...) where T <: MaybeHotVector = _typed_hcat(T, xs)
Base.hcat(xs::MaybeHotVector...) = _typed_hcat(_promote_types(xs...), xs)
function _typed_hcat(::Type{MaybeHotVector{T, U, V}}, xs::Tuple{Vararg{MaybeHotVector}}) where {T, U, V}
    MaybeHotMatrix{T, U, V}([x.i for x in xs], _check_l(xs))
end

Base.reduce(::typeof(hcat), xs::Vector{<:MaybeHotVector}) = _typed_hcat(mapreduce(typeof, promote_type, xs), xs)
function _typed_hcat(::Type{MaybeHotVector{T, U, V}}, xs::AbstractVector{<:MaybeHotVector}) where {T, U, V}
    MaybeHotMatrix{T, U, V}([x.i for x in xs], _check_l(xs))
end

reduce(::typeof(catobs), as::Vector{<:MaybeHotVector}) = reduce(hcat, as)

A::AbstractMatrix * b::MaybeHotVector = (_check_mul(A, b); _mul(A, b))
Zygote.@adjoint A::AbstractMatrix * b::MaybeHotVector = (_check_mul(A, b); Zygote.pullback(_mul, A, b))

_mul(A::AbstractMatrix, b::MaybeHotVector{Missing}) = fill(missing, size(A, 1))
_mul(A::AbstractMatrix, b::MaybeHotVector{<:Integer}) = A[:, b.i]

# this is a bit shady because we're overloading unexported method not intended for public use
Flux._fast_argmax(x::MaybeHotVector) = x.i

Flux.onehot(x::MaybeHotVector{<:Integer}) = Flux.onehot(x.i, 1:x.l)
maybecold(x::MaybeHotVector{Missing}, labels = 1:length(x)) = missing
maybecold(x::MaybeHotVector{<:Integer}, labels = 1:length(x)) = labels[x.i]

"""
    maybehot(l, labels)

Return a [`MaybeHotVector`](@ref) where the first occurence of `l` in `labels` is set to `1`
and all other elements are set to `0`.

# Examples
```jldoctest
julia> maybehot(:b, [:a, :b, :c])
3-element MaybeHotVector with eltype Bool:
 ⋅
 1
 ⋅

julia> maybehot(missing, 1:3)
3-element MaybeHotVector with eltype Missing:
 missing
 missing
 missing
```

See also: [`maybehotbatch`](@ref), [`MaybeHotVector`](@ref), [`MaybeHotMatrix`](@ref).
"""
maybehot(::Missing, labels) = MaybeHotVector(missing, length(labels))
function maybehot(l, labels)
    i = findfirst(isequal(l), labels)
    isnothing(i) && ArgumentError("Value $l not in labels $labels") |> throw
    MaybeHotVector(UInt32(i), length(labels))
end

Base.hash(x::MaybeHotVector, h::UInt) = hash((x.i, x.l), h)
(x1::MaybeHotVector == x2::MaybeHotVector) = x1.i == x2.i && x1.l == x2.l
Base.isequal(x1::MaybeHotVector, x2::MaybeHotVector) = isequal(x1.i, x2.i) && x1.l == x2.l
