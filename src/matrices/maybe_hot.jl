struct MaybeHotVector{T <: Maybe{Integer}} <: AbstractVector{T}
    i::T
    l::Int
end

Base.size(x::MaybeHotVector) = (x.l,)
Base.length(x::MaybeHotVector) = x.l
Base.getindex(x::MaybeHotVector, i::Integer) = (@boundscheck checkbounds(x, i); x.i == i)
Base.getindex(x::MaybeHotVector, ::Colon) = MaybeHotVector(x.i, x.l)

Base.hcat(x::MaybeHotVector) = x
Base.hcat(x::MaybeHotVector, xs::MaybeHotVector...) = hcat((MaybeHotMatrix([x.i], x.l) for x in [x, xs...])...)

function Base.:*(A::AbstractMatrix, b::MaybeHotVector)
    if size(A, 2) != size(b, 1)
        DimensionMismatch(
            "Number of columns of A ($(size(A, 2))) must correspond with length of b ($(length(b)))"
        ) |> throw
    end
    ismissing(b.i) ? fill(missing, size(A, 1)) : A[:, b.i]
end

Base.hash(x::MaybeHotVector{T}, h::UInt) where {T} = hash((T, x.i, x.l), h)
(x1::MaybeHotVector == x2::MaybeHotVector) = isequal(x1.i, x2.i) && x1.l == x2.l

struct MaybeHotMatrix{T <: Maybe{Integer}, V <: AbstractVector{T}} <: AbstractMatrix{T}
    I::V
    l::Int
end

Base.size(X::MaybeHotMatrix) = (X.l, length(X.I))
Base.length(X::MaybeHotMatrix) = X.l * length(X.I)
function Base.getindex(X::MaybeHotMatrix, idcs...)
    @boundscheck checkbounds(X, idcs...)
    _getindex(X, idcs...)
end
_getindex(X::MaybeHotMatrix, i::Union{Integer, AbstractVector}, j::Integer) = X.I[j] .== i
_getindex(X::MaybeHotMatrix, i::Integer, ::Colon) = X.I .== i
_getindex(X::MaybeHotMatrix, idcs::CartesianIndex{2}) = _getindex(X, Tuple(idcs)...)
_getindex(X::MaybeHotMatrix, ::Colon, i::Integer) = MaybeHotVector(X.I[i], X.l)
_getindex(X::MaybeHotMatrix, ::Colon, i::AbstractArray) = MaybeHotMatrix(X.I[i], X.l)
_getindex(X::MaybeHotMatrix, ::Colon, ::Colon) = MaybeHotMatrix(copy(X.I), X.l)

function Base.hcat(X::MaybeHotMatrix, Xs::MaybeHotMatrix...)
    ls = unique(vcat(X.l, [X.l for X in Xs])) 
    if length(ls) > 1
        DimensionMismatch(
            "Number of rows of MaybeHot to hcat must correspond"
        ) |> throw
    end
    MaybeHotMatrix(vcat(X.I, [X.I for X in Xs]...), only(ls))
end

_mul(A::AbstractMatrix, B::MaybeHotMatrix) = hcat((A * MaybeHotVector(i, B.l) for i in B.I)...)
_mul(A::AbstractMatrix, B::MaybeHotMatrix{<:Integer}) = A[:, B.I]
function Base.:*(A::AbstractMatrix, B::MaybeHotMatrix)
    if size(A, 2) != size(B, 1)
        DimensionMismatch(
            "Number of columns of A ($(size(A, 2))) must correspond with number of rows of B ($(size(B, 1)))"
        ) |> throw
    end
    _mul(A, B)
end

Base.hash(X::MaybeHotMatrix{T, V}, h::UInt) where {T, V} = hash((T, V, X.I, X.l), h)
(X1::MaybeHotMatrix == X2::MaybeHotMatrix) = isequal(X1.I, X2.I) && X1.l == X2.l
