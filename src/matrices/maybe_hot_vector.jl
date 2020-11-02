struct MaybeHotVector{T, U, V} <: AbstractVector{V}
    i::T
    l::U

    MaybeHotVector(i::T, l::U) where {T <: Integer, U <: Integer} = new{T, U, Bool}(i, l)
    MaybeHotVector(i::T, l::U) where {T <: Missing, U <: Integer} = new{T, U, Missing}(i, l)
end

Base.size(x::MaybeHotVector) = (x.l,)
Base.length(x::MaybeHotVector) = x.l
Base.getindex(x::MaybeHotVector, i::Integer) = (@boundscheck checkbounds(x, i); x.i == i)
Base.getindex(x::MaybeHotVector, ::Colon) = MaybeHotVector(x.i, x.l)

# we leave the vector type
Base.hcat(x::MaybeHotVector) = x
Base.hcat(xs::MaybeHotVector...) = reduce(hcat, collect(xs))
function Base.reduce(::typeof(hcat), xs::Vector{<:MaybeHotVector})
    reduce(hcat,  MaybeHotMatrix.(xs))
end

Base.reduce(::typeof(catobs), xs::Vector{<:MaybeHotVector}) = reduce(hcat, xs)

A::AbstractMatrix * b::MaybeHotVector = (_check_mul(A, b); _mul(A, b))
Zygote.@adjoint A::AbstractMatrix * b::MaybeHotVector = (_check_mul(A, b); Zygote.pullback(_mul, A, b))

_mul(A::AbstractMatrix, b::MaybeHotVector{Missing}) = fill(missing, size(A, 1))
_mul(A::AbstractMatrix, b::MaybeHotVector{<:Integer}) = A[:, b.i]

Base.hash(x::MaybeHotVector{T, U, V}, h::UInt) where {T, U, V} = hash((T, U, V, x.i, x.l), h)
(x1::MaybeHotVector == x2::MaybeHotVector) = isequal(x1.i, x2.i) && x1.l == x2.l

Flux.onehot(x::MaybeHotVector{<:Integer}) = onehot(x.i, 1:x.l)
