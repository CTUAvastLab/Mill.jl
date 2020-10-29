struct MaybeHotVector{T <: MissingElement{Integer}} <: AbstractVector{Bool}
    i::T
    l::Int
end

Base.size(x::MaybeHotVector) = (x.l,)
Base.length(x::MaybeHotVector) = x.l
Base.getindex(x::MaybeHotVector, i::Integer) = (@boundscheck checkbounds(x, i); x.i == i)
Base.getindex(x::MaybeHotVector, ::Colon) = MaybeHotVector(x.i, x.l)

function Base.:*(A::AbstractMatrix, b::MaybeHotVector)
    if size(A, 2) != size(b, 1)
        DimensionMismatch(
            "Number of columns of A ($(size(A, 2))) must correspond with length of b ($(length(b)))"
        ) |> throw
    end
    ismissing(b.i) ? fill(missing, size(A, 1)) : A[:, b.i]
end

Base.hash(x::MaybeHotVector{T}, h::UInt) where {T} = hash((T, x.i, x.l), h)
(x1::MaybeHotVector{T} == x2::MaybeHotVector{T}) where {T} = x1.i == x2.i && x1.l == x2.l

struct MaybeHotMatrix{T <: MissingElement{Integer}, V <: AbstractVector{T}} <: AbstractMatrix{Bool}
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
_getindex(X::MaybeHotMatrix, ::Colon, i::Integer) = MaybeHotVector(X.I[i], X.l)
_getindex(X::MaybeHotMatrix, ::Colon, i::AbstractArray) = MaybeHotMatrix(X.I[i], X.l)
_getindex(X::MaybeHotMatrix, ::Colon, ::Colon) = MaybeHotMatrix(copy(X.I), X.l)
_getindex(X::MaybeHotMatrix, i::Integer, ::Colon) = map(X -> X[i], X.I)

function Base.hcat(X::MaybeHotMatrix, Ys::MaybeHotMatrix...)
    ls = unique(vcat(X.l, [Y.l for Y in Ys])) 
    if length(ls) > 1
        DimensionMismatch(
            "Number of rows of matrix to hcat must correspond"
        ) |> throw
    end
    MaybeHotMatrix(vcat(X.I, [Y.I for Y in Ys]), only(ls))
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

Base.hash(X::MaybeHotMatrix{T}, h::UInt) where {T} = hash((T, X.I, X.l), h)
(X1::MaybeHotMatrix{T} == X2::MaybeHotMatrix{T}) where {T} = X1.I == X2.I && X1.l == X2.l
