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
end

NGramIterator(s::AbstractString, n::Int=3, b::Int=256) = NGramIterator(codeunits(s), n, b)

Base.length(it::NGramIterator) = length(it.s) + it.n - 1

function Base.iterate(it::NGramIterator, s = (0, 1))
    idx, i = s
    b, n = it.b, it.n
    if i <= length(it.s)
        idx = idx * b + it.s[i]
        idx = i > n ? idx - it.s[i - n] * b^n : idx
        return idx, (idx, i + 1)
    elseif i < length(it.s) + n
        idx = i > n ? idx - it.s[i - n] * b^(n - (i - length(it.s))) : idx
        return idx, (idx, i + 1)
    else
        return nothing
    end
end

"""
ngrams!(o,x,n::Int,b::Int)

store indexes of `n` grams of `x` with base `b` to `o`

"""
function ngrams!(o, x::T, n::Int, b::Int) where {T<:Union{CodeUnits{UInt8}, Vector{<:Integer}}}
    @assert length(o) >= length(x) + n - 1
    for (i, idx) in enumerate(NGramIterator(x, n, b))
        o[i] = idx
    end
    o
end

"""
ngrams(x,n::Int,b::Int)

indexes of `n` grams of `x` with base `b`

"""
ngrams(x::Union{CodeUnits{UInt8}, Vector{<:Integer}}, n::Int, b::Int) = collect(NGramIterator(x, n, b))
ngrams(x::AbstractString, n::Int, b::Int) = ngrams(codeunits(x),n,b)

"""
function countngrams!(o,x,n::Int,b::Int)

counts number of of `n` grams of `x` with base `b` to `o` and store it to o

"""
function countngrams!(o, x::Union{CodeUnits{UInt8},Vector{<:Integer}}, n::Int, b::Int)
    for idx in NGramIterator(x, n, b)
        o[mod(idx, length(o)) + 1] += 1
    end
    o
end

countngrams!(o, x::AbstractString, n::Int, b::Int) = countngrams!(o, codeunits(x), n, b)

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

string2ngrams(x::AbstractArray{<:AbstractString}, n, m) = countngrams(Vector(x[:]), n, 256, m)
string2ngrams(x::AbstractString, n, m) = countngrams(x, n, 256, m)
string2ngrams(x, n, m) = x

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
# TODO use better type here
struct NGramMatrix{T} <: AbstractMatrix{T}
    s::Vector{T}
    n::Int
    b::Int
    m::Int
end

NGramMatrix(s::Vector{T}, n::Int=3, b::Int=256, m::Int=2053) where T <: AbstractString = NGramMatrix{T}(s, n, b, m)
NGramMatrix(s::AbstractString, args...) = NGramMatrix([s], args...)

function Base.show(io::IO, X::NGramMatrix)
    if get(io, :compact, false)
        print(io, size(X, 1), "x", size(X, 2), " NGramMatrix")
    else
        print(io, "NGramMatrix(s = ", X.s, ", n = ", X.n, 
              ", b = ", X.b, ", m = ", X.m, ")")
    end
end

print_array(io::IO, A::NGramMatrix) = print_array(io, A.s)

NGramIterator(a::NGramMatrix{<:AbstractString}, i::Integer) = NGramIterator(codeunits(a.s[i]), a.n, a.b)
NGramIterator(a::NGramMatrix{<:Vector{<:Integer}}, i::Integer) = NGramIterator(a.s[i], a.n, a.b)

Base.length(a::NGramMatrix) = a.m * length(a.s)
Base.size(a::NGramMatrix) = (a.m, length(a.s))
Base.size(a::NGramMatrix, d) = (d == 1) ? a.m : length(a.s)
Base.getindex(a::NGramMatrix{<:AbstractString}, ::Colon, i::Integer) = NGramMatrix([a.s[i]], a.n, a.b, a.m)
Base.getindex(a::NGramMatrix{<:AbstractString}, ::Colon, i::AbstractArray) = NGramMatrix(a.s[i], a.n, a.b, a.m)

subset(a::NGramMatrix{<:AbstractString}, i) = NGramMatrix(a.s[i], a.n, a.b, a.m)

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

Base.Matrix(x::NGramMatrix) = Matrix(SparseMatrixCSC(x))

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
            I[vid] = mod(i, x.m) + 1
            J[vid] = j
            vid += 1
        end
    end
    sparse(I, J, V, size(x,1), size(x,2))
end

function nextgram(s, idx, i, N, B)
    if i <= length(s)
        idx = idx * B + s[i]
        idx = (i>N) ? idx - s[i - N]*B^N : idx
        return(idx)
       end
    idx = (i>N) ? idx - s[i - N]*B^(N - (i - length(s))) : idx
    idx
end

function mulkernel!(C, A, jB, mA, nA, idxs)
    for iB in idxs
        miB = mod(iB, nA) + 1
        @inbounds for iA in 1:mA
            C[iA, jB] += A[iA, miB]
        end
    end
end

function mulkernel!(C, A, jB, mA, nA, s, N, B)
    l = length(s) + N - 1
    iB = 0
    for j in 1:l
        iB = nextgram(s, iB, j, N, B)
        miB = mod(iB, nA) + 1
        @inbounds for iA in 1:mA
            C[iA, jB] += A[iA, miB]
        end
    end
end

*(A::Matrix, B::NGramMatrix) = mul(A, B)
function mul(A::AbstractMatrix, B::NGramMatrix{T}) where {T<:AbstractString}
    mA, nA = size(A)
    @assert nA == size(B,1)
    C = zeros(eltype(A), mA, size(B, 2))
    for jB in 1:size(B, 2)
        # mulkernel!(C, A, jB, mA, nA, NGramIterator(B, jB))
        mulkernel!(C, A, jB, mA, nA, codeunits(B.s[jB]), B.n, B.b)
    end
    return C
end

function mul(A::AbstractMatrix, B::NGramMatrix{T}) where {T}
    mA, nA = size(A)
    @assert nA == size(B,1)
    C = zeros(eltype(A), mA, size(B, 2))
    for jB in 1:size(B, 2)
        # mulkernel!(C, A, jB, mA, nA, NGramIterator(B, jB))
        mulkernel!(C, A, jB, mA, nA, B.s[jB], B.n, B.b)
    end
    return C
end

function multkernel!(C, A, jB, mA, bm, idxs)
    for iB in idxs
        miB = mod(iB, bm) + 1
        @inbounds for iA in 1:mA
            C[iA, miB] += A[iA, jB]
        end
    end
end

function multkernel!(C, A, jB, mA, bm, s, N, B)
    l = length(s) + N - 1
    iB = 0
    for j in 1:l
        iB = nextgram(s, iB, j, N, B)
        miB = mod(iB, bm) + 1
        @inbounds for iA in 1:mA
            C[iA, miB] += A[iA, jB]
        end
    end
end

function multrans(A::AbstractMatrix, B::NGramMatrix{T}) where {T<:AbstractString}
    mA, nA = size(A)
    C = zeros(eltype(A), mA, B.m)
    for jB in 1:size(B, 2)
        # multkernel!(C, A, jB, mA, B.m, NGramIterator(B, jB))
        multkernel!(C, A, jB, mA, B.m, codeunits(B.s[jB]), B.n, B.b)
    end
    return C
end

function multrans(A::AbstractMatrix, B::NGramMatrix{T}) where {T}
    mA, nA = size(A)
    C = zeros(eltype(A), mA, B.m)
    for jB in 1:size(B, 2)
        # multkernel!(C, A, jB, mA, B.m, NGramIterator(B, jB))
        multkernel!(C, A, jB, mA, B.m, B.s[jB], B.n, B.b)
    end
    return C
end

# TODO change to rrule once this gets resolved
# https://github.com/FluxML/Zygote.jl/issues/811
# function rrule(::typeof(mul), a::AbstractMatrix, b::NGramMatrix)
#     return mul(a, b), Δ -> (NO_FIELDS, multrans(Δ, b), DoesNotExist())
# end
Zygote.@adjoint function mul(a::AbstractMatrix, b::NGramMatrix)
    return mul(a,b) , Δ -> (multrans(Δ, b), nothing)
end
# function rrule(::typeof(*), a::AbstractMatrix, b::NGramMatrix)
#     return mul(a, b), Δ -> (NO_FIELDS, multrans(Δ, b), DoesNotExist())
# end
Zygote.@adjoint function *(a::AbstractMatrix, b::NGramMatrix)
    return mul(a,b) , Δ -> (multrans(Δ, b), nothing)
end

Base.hash(e::NGramMatrix{T}, h::UInt) where {T} = hash((T, e.s, e.n, e.b, e.m), h)
(e1::NGramMatrix{T} == e2::NGramMatrix{T}) where {T} = e1.s == e2.s && e1.n === e2.n && e1.b === e2.b && e1.m === e2.m
