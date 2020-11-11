const Code = Union{AbstractVector{<:Integer}, CodeUnits}
const Sequence = Union{AbstractString, Code}

"""
struct NGramIterator{T}
s::T
n::Int
b::Int
end

Iterates and enumerates ngrams of collection of integers `s::T` with zero padding. Enumeration is computed as in positional number systems, where items of `s` are digits and `b` is the base.

In order to reduce collisions when mixing ngrams of different order one should avoid zeros and negative integers in `s` and should set base `b` to be equal to the expected number of unique tokkens in `s`.

# Examples
```jldoctest
julia> it = Mill.NGramIterator(collect(1:10), 3, 10)
Mill.NGramIterator{Array{Int64,1}}([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, 10)

julia> collect(it)
12-element Array{Int64,1}:
1
12
123
234
345
456
567
678
789
900
100
10

julia> sit = Mill.NGramIterator(codeunits("deadbeef"), 3, 256)    # creates collisions as codeunits returns tokens from 0x00:0xff
Mill.NGramIterator{Base.CodeUnits{UInt8,String}}(UInt8[0x64, 0x65, 0x61, 0x64, 0x62, 0x65, 0x65, 0x66], 3, 256)

julia> collect(sit)
10-element Array{Int64,1}:
100
25701
6579553
6644068
6382690
6578789
6448485
6645094
25958
102
```
"""
struct NGramIterator{T}
    s::T
    n::Int
    b::Int
    m::Int

    NGramIterator(s::T, n::Int=3, b::Int=256, m::Int=typemax(Int)) where T <: Maybe{Code} = new{T}(s, n, b, m)
end

NGramIterator(s::AbstractString, args...) = NGramIterator(codeunits(s), args...)

Base.eltype(::NGramIterator) = Int

Base.length(it::NGramIterator) = length(it.s) + it.n - 1
Base.length(it::NGramIterator{Missing}) = 0

Base.iterate(it::NGramIterator{Missing}) = nothing
function Base.iterate(it::NGramIterator, (z, i) = (_init_z(it), 1))
    b, n, s, m = it.b, it.n, it.s, it.m
    if i ≤ length(it)
        z *= b
        z += i ≤ length(s) ? s[i] : string_end_code()
        z -= b^n * (i > n ? s[i - n] : string_start_code())
        z%m, (z, i+1)
    end
end

_init_z(it) = string_start_code() * foldl((s, c) -> it.b * s + c, fill(1, it.n))

Base.hash(it::NGramIterator{T}, h::UInt) where {T} = hash((T, it.s, it.n, it.b, it.m), h)
(it1::NGramIterator{T} == it2::NGramIterator{T}) where {T} = it1.s == it2.s &&
    it1.n === it2.n && it1.b === it2.b && it1.m === it2.m

"""
ngrams!(o,x,n::Int,b::Int)

store indexes of `n` grams of `x` with base `b` to `o`

"""
function ngrams!(o, x::Sequence, n::Int, b::Int)
    for (i, idx) in enumerate(NGramIterator(x, n, b))
        o[i] = idx
    end
    o
end

"""
ngrams(x,n::Int,b::Int)

indexes of `n` grams of `x` with base `b`

"""
ngrams(x::Sequence, n::Int, b::Int) = collect(NGramIterator(x, n, b))

"""
function countngrams!(o,x,n::Int,b::Int)

counts number of of `n` grams of `x` with base `b` to `o` and store it to o

"""
countngrams!(o, x::Sequence, n::Int, b::Int) = countngrams!(o, x, n, b, length(o))
function countngrams!(o, x::Sequence, n::Int, b::Int, m::Int)
    for idx in NGramIterator(x, n, b, m)
        o[idx + 1] += 1
    end
    o
end

"""
function countngrams(x,n::Int,b::Int)

counts number of of `n` grams of `x` with base `b` to `o`

"""
countngrams(x, n::Int, b::Int, m) = countngrams!(zeros(Int,m), x, n, b)
function countngrams(x::Vector{<:AbstractString}, n::Int, b::Int, m)
    o = zeros(Int, m, length(x))
    for (i,s) in enumerate(x)
        countngrams!(view(o,:,i), x[i], n, b)
    end
    o
end

string2ngrams(x::AbstractArray{<:AbstractString}, n, b, m) = countngrams(Vector(x[:]), n, b, m)
string2ngrams(x::AbstractString, n, b, m) = countngrams(x, n, b, m)

"""
struct NGramMatrix{T}
s::Vector{T}
n::Int
b::Int
m::Int
end

Represents strings stored in array `s` as ngrams of cardinality `n`. Strings are internally stored as strings and the multiplication with
dense matrix is overloaded and `b` is a base for calculation of trigrams. Finally `m` is the modulo applied on indexes of ngrams.

The structure essentially represents module one-hot representation of strings, where each columns contains one observation (string).
Therefore the structure can be viewed as a matrix with `m` rows and `length(s)` columns
"""
struct NGramMatrix{T, U <: AbstractVector{T}, V} <: AbstractMatrix{V}
    s::U
    n::Int
    b::Int
    m::Int

    function NGramMatrix(s::U, n::Int=3, b::Int=256, m::Int=2053) where {T <: Sequence, U <: AbstractVector{T}}
        new{T, U, Int}(s, n, b, m)
    end

    NGramMatrix(s::U, n::Int=3, b::Int=256, m::Int=2053) where U <: AbstractVector{Missing} =
        new{Missing, U, Missing}(s, n, b, m)

    function NGramMatrix(s::U, n::Int=3, b::Int=256, m::Int=2053) where {T <: Maybe{Sequence}, U <: AbstractVector{T}}
        new{T, U, Union{Missing, Int}}(s, n, b, m)
    end
end

NGramMatrix(s::Maybe{Sequence}, args...) = NGramMatrix([s], args...)

NGramIterator(A::NGramMatrix, i::Integer) = NGramIterator(A.s[i], A.n, A.b, A.m)

Base.length(A::NGramMatrix) = A.m * length(A.s)
Base.size(A::NGramMatrix) = (A.m, length(A.s))
Base.size(A::NGramMatrix, d) = (d == 1) ? A.m : length(A.s)

Base.getindex(X::NGramMatrix, idcs...) = (@boundscheck checkbounds(X, idcs...); _getindex(X, idcs...))
_getindex(X::NGramMatrix{<:AbstractString}, ::Colon, i::Integer) = NGramMatrix([X.s[i]], X.n, X.b, X.m)
_getindex(X::NGramMatrix{<:AbstractString}, ::Colon, i::AbstractArray) = NGramMatrix(X.s[i], X.n, X.b, X.m)

subset(a::NGramMatrix, i) = NGramMatrix(a.s[i], a.n, a.b, a.m)

Base.reduce(::typeof(catobs), As::Vector{<:NGramMatrix}) = reduce(hcat, As)
Base.hcat(As::NGramMatrix...) = reduce(hcat, collect(As))
function Base.reduce(::typeof(hcat), As::Vector{<:NGramMatrix})
    ns = unique([A.n for A in As])
    bs = unique([A.b for A in As])
    ms = unique([A.m for A in As])
    if length(ns) > 1 || length(bs) > 1 || length(ms) > 1
        DimensionMismatch(
            "Matrices do not have the same n, b, or m."
        ) |> throw
    end
    NGramMatrix(reduce(vcat, [i.s for i in As]), only(ns), only(bs), only(ms))
end

SparseArrays.SparseMatrixCSC(x::NGramMatrix) = SparseArrays.SparseMatrixCSC{Float32, UInt}(x)
function SparseArrays.SparseMatrixCSC{Tv, Ti}(x::NGramMatrix) where {Tv, Ti <: Integer}
    size(x, 2) == 0 && return sparse(Ti[],Ti[],Tv[], size(x,1), size(x,2))
    l = sum(map(i -> length(NGramIterator(x, i)), 1:size(x, 2)))
    I = zeros(Ti, l)
    J = zeros(Ti, l)
    V = ones(Tv, l)
    vid = 1
    for j in 1:size(x, 2)
        for i in NGramIterator(x, j)
            I[vid] = i + 1
            J[vid] = j
            vid += 1
        end
    end
    sparse(I, J, V, size(x,1), size(x,2))
end

A::AbstractMatrix * B::NGramMatrix = (_check_mul(A, B); _mul(A, B))
Zygote.@adjoint A::AbstractMatrix * B::NGramMatrix = (_check_mul(A, B); Zygote.pullback(_mul, A, B))

_mul(A::AbstractMatrix, B::NGramMatrix{Missing}) = fill(missing, size(A, 1), size(B, 2))

# TODO rewrite this to less parameters once Zygote allows for composite grads
_mul(A::AbstractMatrix, B::NGramMatrix{T}) where T <: Maybe{Sequence} = _mul(A, B.s, B.n, B.b, B.m)

function _mul(A::AbstractMatrix, S::AbstractVector{T}, n, b, m) where T <: Maybe{Sequence}
    T_res = Missing <: T ? Union{eltype(A), T} : eltype(A)
    C = zeros(T_res, size(A, 1), length(S))
    for (k, s) in enumerate(S)
        _mul_vec!(view(C, :, k), A, NGramIterator(s, n, b, m))
    end
    C
end
Zygote.@adjoint function _mul(A::AbstractMatrix, S::AbstractVector{<:Sequence}, n, b, m)
    function dA_thunk(Δ)
        dA = zero(A)
        for (k, s) in enumerate(S)
            _dA_mul_vec!(view(Δ, :, k), dA, NGramIterator(s, n, b, m))
        end
        dA
    end
    return _mul(A, S, n, b, m), Δ -> (dA_thunk(Δ), nothing, nothing, nothing, nothing)
end

_mul_vec!(c, A, it::NGramIterator, ψ=missing) = for j in it
    @views c .+= A[:, j + 1]
end
_mul_vec!(c, A, it::NGramIterator{Missing}, ψ=missing) = c .= ψ 
_dA_mul_vec!(δ, dA, it::NGramIterator) = for j in it
    @views dA[:, j + 1] .+= δ
end
_dA_mul_vec!(δ, dA, it::NGramIterator{Missing}) = return

Base.hash(e::NGramMatrix{T}, h::UInt) where {T} = hash((T, e.s, e.n, e.b, e.m), h)
(e1::NGramMatrix{T} == e2::NGramMatrix{T}) where {T} = e1.s == e2.s && e1.n === e2.n && e1.b === e2.b && e1.m === e2.m
