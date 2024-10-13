"""
    MaybeHotMatrix{T, V} <: AbstractMatrix{U}

A matrix-like structure for representing one-hot encoded variables.
Like `Flux.OneHotMatrix` but supports `missing` values.

Construct with the [`maybehotbatch`](@ref) function.

See also: [`MaybeHotVector`](@ref), [`maybehot`](@ref).
"""
struct MaybeHotMatrix{T, U} <: AbstractMatrix{U}
    I::Vector{T}
    l::Int
    MaybeHotMatrix(I::Vector{T}, l::Int) where T <: Integer = new{T, Bool}(I, l)
    MaybeHotMatrix(I::Vector{Missing}, l::Int) = new{Missing, Missing}(I, l)
    MaybeHotMatrix(I::Vector{T}, l::Int) where T <: Maybe{Integer} = new{T, Maybe{Bool}}(I, l)

    function MaybeHotMatrix{T, U}(I, l) where {T <: Maybe{Integer}, U <: Maybe{Bool}}
        new{T, U}(convert(Vector{T}, I), l)
    end
end

MaybeHotMatrix(i::Integer, l::Integer) = MaybeHotMatrix([i], l)
MaybeHotMatrix(x::MaybeHotVector) = MaybeHotMatrix([x.i], x.l)

Base.size(X::MaybeHotMatrix) = (X.l, length(X.I))
Base.length(X::MaybeHotMatrix) = X.l * length(X.I)

Base.getindex(X::MaybeHotMatrix, idcs...) = (@boundscheck checkbounds(X, idcs...); _getindex(X, idcs...))
_getindex(X::MaybeHotMatrix, i::Union{Integer, AbstractVector}, j::Integer) = X.I[j] .== i
_getindex(X::MaybeHotMatrix, i::Integer, ::Colon) = X.I .== i
_getindex(X::MaybeHotMatrix, idcs::CartesianIndex{2}) = _getindex(X, Tuple(idcs)...)
_getindex(X::MaybeHotMatrix, ::Colon, i::Integer) = MaybeHotVector(X.I[i], X.l)
_getindex(X::MaybeHotMatrix, ::Colon, i::AbstractArray) = MaybeHotMatrix(X.I[i], X.l)
_getindex(X::MaybeHotMatrix, ::Colon, ::Colon) = MaybeHotMatrix(copy(X.I), X.l)

function Base.convert(::Type{<:MaybeHotMatrix{T}}, X::MaybeHotMatrix) where T
    MaybeHotMatrix(convert(Vector{T}, X.I), X.l)
end

function Base.promote_rule(::Type{MaybeHotMatrix{T, U}},
        ::Type{MaybeHotMatrix{A, B}}) where {T, A, U, B}
    MaybeHotMatrix{promote_type(T, A), promote_type(U, B)}
end

function _check_l(Xs::AbstractVecOrTuple{MaybeHotMatrix})
    l = Xs[1].l
    if any(!isequal(l), (X.l for X in Xs))
        DimensionMismatch(
            "Number of rows of `MaybeHotMatrix`s to hcat must correspond"
        ) |> throw
    end
    l
end

Base.hcat(Xs::T...) where T <: MaybeHotMatrix = _typed_hcat(T, Xs)
Base.hcat(Xs::MaybeHotMatrix...) = _typed_hcat(_promote_types(Xs...), Xs)
function _typed_hcat(::Type{T}, Xs::Tuple{Vararg{MaybeHotMatrix}}) where T <: MaybeHotMatrix
    T(vcat([X.I for X in Xs]...), _check_l(Xs))
end

Base.reduce(::typeof(hcat), Xs::Vector{<:MaybeHotMatrix}) = _typed_hcat(mapreduce(typeof, promote_type, Xs), Xs)
function _typed_hcat(::Type{T}, Xs::AbstractVector{<:MaybeHotMatrix}) where T <: MaybeHotMatrix
    T(reduce(vcat, [X.I for X in Xs]), _check_l(Xs))
end

A::AbstractMatrix * B::MaybeHotMatrix = (_check_mul(A, B); _mul(A, B))

_mul(A::AbstractMatrix, B::MaybeHotMatrix{Missing}) = fill(missing, size(A, 1), size(B, 2))
_mul(A::AbstractMatrix, B::MaybeHotMatrix{<:Integer}) = A[:, B.I]
function _mul(A::AbstractMatrix, B::MaybeHotMatrix)
    C = zeros(Maybe{eltype(A)}, size(A, 1), size(B, 2))
    @inbounds for (k,i) in enumerate(B.I)
        C[:, k] = A * MaybeHotVector(i, B.l)
    end
    C
end

# this is a bit shady because we're overloading unexported method not intended for public use
OneHotArrays._fast_argmax(X::MaybeHotMatrix) = X.I

Flux.onehotbatch(X::MaybeHotMatrix{<:Integer}) = Flux.onehotbatch(X.I, 1:X.l)

"""
    maybehotbatch(ls, labels)

Return a [`MaybeHotMatrix`](@ref) in which each column corresponds to one element of `ls`
containing `1` at its first occurence in `labels` with all other elements set to `0`.

# Examples
```jldoctest
julia> maybehotbatch([:c, :a], [:a, :b, :c])
3×2 MaybeHotMatrix with eltype Bool:
 ⋅  1
 ⋅  ⋅
 1  ⋅

julia> maybehotbatch([missing, 2], 1:3)
3×2 MaybeHotMatrix with eltype Union{Missing, Bool}:
 missing  ⋅
 missing  1
 missing  ⋅
```

See also: [`maybehot`](@ref), [`MaybeHotMatrix`](@ref), [`MaybeHotVector`](@ref).
"""
maybehotbatch(L, labels) = MaybeHotMatrix([maybehot(l, labels).i for l in L], length(labels))

function maybecold(X::MaybeHotMatrix{<:Maybe{Integer}}, labels=1:size(X, 1))
    indices = OneHotArrays._fast_argmax(X)
    xs = isbits(labels) ? indices : collect(indices) # non-bit type cannot be handled by CUDA
    return map(xi -> ismissing(xi) ? xi : labels[xi[1]], xs)
end
maybecold(X::MaybeHotMatrix{<:Integer}, labels=1:size(X, 1)) = Flux.onecold(X, labels)

Base.hash(X::MaybeHotMatrix, h::UInt) = hash((X.I, X.l), h)
(X1::MaybeHotMatrix == X2::MaybeHotMatrix) = X1.I == X2.I && X1.l == X2.l
Base.isequal(X1::MaybeHotMatrix, X2::MaybeHotMatrix) = isequal(X1.I, X2.I) && X1.l == X2.l
