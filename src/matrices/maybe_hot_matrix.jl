struct MaybeHotMatrix{T, U, V, W} <: AbstractMatrix{W}
    I::U
    l::V
    MaybeHotMatrix(I::U, l::V) where {T <: Integer, U <: AbstractVector{T}, V <: Integer} = new{T, U, V, Bool}(I, l)
    MaybeHotMatrix(I::U, l::V) where {U <: AbstractVector{Missing}, V <: Integer} = new{Missing, U, V, Missing}(I, l)
    function MaybeHotMatrix(I::U, l::V) where {T <: Maybe{Integer}, U <: AbstractVector{T}, V <: Integer}
        new{T, U, V, Union{Bool, Missing}}(I, l)
    end
end

MaybeHotMatrix(x::MaybeHotVector) = MaybeHotMatrix([x.i], x.l)
MaybeHotMatrix(i::Integer, l::Integer) = MaybeHotMatrix([i], l)

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

Base.hcat(Xs::MaybeHotMatrix...) = reduce(hcat, collect(Xs))
function Base.reduce(::typeof(hcat), Xs::Vector{<:MaybeHotMatrix})
    ls = unique([X.l for X in Xs])
    if length(ls) > 1
        DimensionMismatch(
            "Number of rows of MaybeHot to hcat must correspond"
        ) |> throw
    end
    MaybeHotMatrix(reduce(vcat, [X.I for X in Xs]), only(ls))
end

Base.reduce(::typeof(catobs), Xs::Vector{<:MaybeHotMatrix}) = reduce(hcat, Xs)

A::AbstractMatrix * B::MaybeHotMatrix = (_check_mul(A, B); _mul(A, B))
Zygote.@adjoint A::AbstractMatrix * B::MaybeHotMatrix = (_check_mul(A, B); Zygote.pullback(_mul, A, B))

_mul(A::AbstractMatrix, B::MaybeHotMatrix) = hcat((A * MaybeHotVector(i, B.l) for i in B.I)...)
_mul(A::AbstractMatrix, B::MaybeHotMatrix{<:Integer}) = A[:, B.I]

Base.hash(X::MaybeHotMatrix{T, U, V, W}, h::UInt) where {T, U, V, W} = hash((T, U, V, W, X.I, X.l), h)
(X1::MaybeHotMatrix == X2::MaybeHotMatrix) = isequal(X1.I, X2.I) && X1.l == X2.l

Flux.onehotbatch(X::MaybeHotMatrix{<:Integer}) = onehotbatch(X.I, 1:X.l)

maybehotbatch(L, labels) = let ohvs = [maybehot(l, labels) for l in L]
    MaybeHotMatrix([ohv.i for ohv in ohvs], length(labels))
end
