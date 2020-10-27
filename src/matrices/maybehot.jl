struct MaybeHotVector{T <: MissingElement{Integer}, U} <: AbstractVector{Bool}
    i::T
    l::U
end

print_array(io::IO, x::MaybeHotVector) = print_array(io, [x.i])

Base.size(x::MaybeHotVector) = (x.l,)
Base.length(x::MaybeHotVector) = x.l
Base.getindex(x::MaybeHotVector, i::Integer) = x.i == i
Base.getindex(x::MaybeHotVector, ::Colon) = MaybeHotVector(x.i, x.l)

struct MaybeHotMatrix{T <: MissingElement{Integer}, U, V <: AbstractVector{T}} <: AbstractMatrix{Bool}
    I::V
    l::U
end

function print_array(io::IO, X::MaybeHotMatrix)
    println(io, "I:")
    print_array(io, X.I)
end

Base.size(X::MaybeHotMatrix) = (X.l, length(X.I))
Base.length(X::MaybeHotMatrix) = X.l * length(X.I)
