struct MaybeHotVector{T <: Maybe{Integer}} <: AbstractVector{T}
    i::T
    l::Int
end

Base.size(x::MaybeHotVector) = (x.l,)
Base.length(x::MaybeHotVector) = x.l
Base.getindex(x::MaybeHotVector, i::Integer) = (@boundscheck checkbounds(x, i); x.i == i)
Base.getindex(x::MaybeHotVector, ::Colon) = MaybeHotVector(x.i, x.l)

Base.hcat(x::MaybeHotVector) = x
Base.hcat(x::MaybeHotVector, xs::MaybeHotVector...) = reduce(hcat, vcat([x], collect(xs)))
function Base.reduce(::typeof(hcat), xs::Vector{<:MaybeHotVector})
    reduce(hcat,  MaybeHotMatrix.(xs))
end

A::AbstractMatrix * b::MaybeHotVector = (_check_mul(A, b); _mul(A, b))
Zygote.@adjoint A::AbstractMatrix * b::MaybeHotVector = (_check_mul(A, b); Zygote.pullback(_mul, A, b))

_mul(A::AbstractMatrix, b::MaybeHotVector{Missing}) = fill(missing, size(A, 1))
_mul(A::AbstractMatrix, b::MaybeHotVector{<:Integer}) = A[:, b.i]

Base.hash(x::MaybeHotVector{T}, h::UInt) where {T} = hash((T, x.i, x.l), h)
(x1::MaybeHotVector == x2::MaybeHotVector) = isequal(x1.i, x2.i) && x1.l == x2.l

Flux.onehot(x::MaybeHotVector{<:Integer}) = onehot(x.i, 1:x.l)
