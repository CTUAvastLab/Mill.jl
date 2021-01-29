struct MaybeHotMatrix{T, U, V, W} <: AbstractMatrix{W}
    I::U
    l::V
    MaybeHotMatrix(I::U, l::V) where {T <: Integer, U <: AbstractVector{T}, V <: Integer} = new{T, U, V, Bool}(I, l)
    MaybeHotMatrix(I::U, l::V) where {U <: AbstractVector{Missing}, V <: Integer} = new{Missing, U, V, Missing}(I, l)
    function MaybeHotMatrix(I::U, l::V) where {T <: Maybe{Integer}, U <: AbstractVector{T}, V <: Integer}
        new{T, U, V, Maybe{Bool}}(I, l)
    end
end

MaybeHotMatrix(x::MaybeHotVector) = MaybeHotMatrix([x.i], x.l)
MaybeHotMatrix(i::Integer, l::Integer) = MaybeHotMatrix([i], l)

Base.size(X::MaybeHotMatrix) = (X.l, length(X.I))
Base.length(X::MaybeHotMatrix) = X.l * length(X.I)

Base.getindex(X::MaybeHotMatrix, idcs...) = (@boundscheck checkbounds(X, idcs...); _getindex(X, idcs...))
_getindex(X::MaybeHotMatrix, i::Union{Integer, AbstractVector}, j::Integer) = X.I[j] .== i
_getindex(X::MaybeHotMatrix, i::Integer, ::Colon) = X.I .== i
_getindex(X::MaybeHotMatrix, idcs::CartesianIndex{2}) = _getindex(X, Tuple(idcs)...)
_getindex(X::MaybeHotMatrix, ::Colon, i::Integer) = MaybeHotVector(X.I[i], X.l)
_getindex(X::MaybeHotMatrix, ::Colon, i::AbstractArray) = MaybeHotMatrix(X.I[i], X.l)
_getindex(X::MaybeHotMatrix, ::Colon, ::Colon) = MaybeHotMatrix(copy(X.I), X.l)

Base.hcat(Xs::MaybeHotMatrix...) = reduce(hcat, collect(Xs))
function Base.reduce(::typeof(hcat), Xs::Vector{MaybeHotMatrix})
    @show "ahoj"
    isempty(Xs) && return Xs
    l = Xs[1].l
    if any(!isequal(l), (X.l for X in Xs))
        DimensionMismatch(
            "Number of rows of MaybeHot to hcat must correspond"
        ) |> throw
    end
    MaybeHotMatrix(reduce(vcat, [X.I for X in Xs]), l)
end

A::AbstractMatrix * B::MaybeHotMatrix = (_check_mul(A, B); _mul(A, B))
Zygote.@adjoint A::AbstractMatrix * B::MaybeHotMatrix = (_check_mul(A, B); Zygote.pullback(_mul, A, B))

_mul(A::AbstractMatrix, B::MaybeHotMatrix{Missing}) = fill(missing, size(A, 1), size(B, 2))
_mul(A::AbstractMatrix, B::MaybeHotMatrix{<:Integer}) = A[:, B.I]
function _mul(A::AbstractMatrix, B::MaybeHotMatrix)
    C = zeros(Union{eltype(A), Missing}, size(A, 1), size(B, 2))
    @inbounds for (k,i) in enumerate(B.I)
        C[:, k] = A * MaybeHotVector(i, B.l)
    end
    C
end

Flux.onehotbatch(X::MaybeHotMatrix{<:Integer}) = onehotbatch(X.I, 1:X.l)

maybehotbatch(L, labels) = MaybeHotMatrix([maybehot(l, labels).i for l in L], length(labels))

Base.hash(X::MaybeHotMatrix, h::UInt) where {T, U, V, W} = hash((X.I, X.l), h)
(X1::MaybeHotMatrix == X2::MaybeHotMatrix) = X1.I == X2.I && X1.l == X2.l
isequal(X1::MaybeHotMatrix, X2::MaybeHotMatrix) = isequal(X1.I, X2.I) && X1.l == X2.l
