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

A::AbstractMatrix * B::MaybeHotMatrix = (_check_mul(A, B); _mul(A, B))
Zygote.@adjoint A::AbstractMatrix * B::MaybeHotMatrix = (_check_mul(A, B); Zygote.pullback(_mul, A, B))

_mul(A::AbstractMatrix, B::MaybeHotMatrix) = hcat((A * MaybeHotVector(i, B.l) for i in B.I)...)
_mul(A::AbstractMatrix, B::MaybeHotMatrix{<:Integer}) = A[:, B.I]

Base.hash(X::MaybeHotMatrix{T, V}, h::UInt) where {T, V} = hash((T, V, X.I, X.l), h)
(X1::MaybeHotMatrix == X2::MaybeHotMatrix) = isequal(X1.I, X2.I) && X1.l == X2.l

Flux.onehotbatch(X::MaybeHotMatrix{<:Integer}) = onehotbatch(X.I, 1:X.l)
