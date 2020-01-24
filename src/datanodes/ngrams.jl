import Base.*
# struct NGramIterator{T} where {T<:Union{Base.CodeUnits{UInt8,S} where S,Vector{I} where I<:Integer}}
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

julia> [gram for gram in it]
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

julia> [gram for gram in sit]
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
    # function NGramIterator{T}(x, n, b) where {T}
    # new(s, n, b)
    # end
end

# NGramIterator(s::AbstractString) = NGramIterator(codeunits(s), 3, 256)
# NGramIterator(s::AbstractString, n, b) = NGramIterator(codeunits(s), n, b)

Base.length(it::NGramIterator) = length(it.s) + it.n - 1

function Base.iterate(it::NGramIterator, s = (0, 1))
    idx, i = s
    b, n = it.b, it.n
    if i <= length(it.s)
        idx = idx * b + it.s[i]
        idx = (i>n) ? idx - it.s[i - n]*b^n : idx
        return(idx, (idx, i + 1))
    elseif i < length(it.s) + n
        idx = (i>n) ? idx - it.s[i - n]*b^(n - (i - length(it.s))) : idx
        return(idx, (idx, i + 1))
    else
        return(nothing)
    end
end

"""
ngrams!(o,x,n::Int,b::Int)

store indexes of `n` grams of `x` with base `b` to `o`

"""
function ngrams!(o,x::T,n::Int,b::Int) where {T<:Union{Base.CodeUnits{UInt8,S} where S,Vector{I} where I<:Integer}}
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
ngrams(x::T,n::Int,b::Int) where {T<:Union{Base.CodeUnits{UInt8,S} where S,Vector{I} where I<:Integer}} =   collect(NGramIterator(x, n, b))
ngrams(x::T,n::Int,b::Int) where {T<:AbstractString} = ngrams(codeunits(x),n,b)

"""
function countngrams!(o,x,n::Int,b::Int)

counts number of of `n` grams of `x` with base `b` to `o` and store it to o

"""
function countngrams!(o,x::T,n::Int,b::Int) where {T<:Union{Base.CodeUnits{UInt8,S} where S,Vector{I} where I<:Integer}}
    for idx in NGramIterator(x, n, b)
        o[mod(idx, length(o))+1] += 1
    end
    o
end

countngrams!(o,x::T,n::Int,b::Int) where {T<:AbstractString} = countngrams!(o,codeunits(x),n,b)

"""
function countngrams(x,n::Int,b::Int)

counts number of of `n` grams of `x` with base `b` to `o`

"""
countngrams(x,n::Int,b::Int,m) = countngrams!(zeros(Int,m),x,n,b)
function countngrams(x::Vector{T},n::Int,b::Int,m) where {T<:AbstractString}
    o = zeros(Int,m,length(x))
    for (i,s) in enumerate(x)
        countngrams!(view(o,:,i),x[i],n,b)
    end
    o
end


string2ngrams(x::T, n, m) where {T <: AbstractArray{I} where I<: AbstractString} = countngrams(Vector(x[:]),n,256,m)
string2ngrams(x::T, n, m) where {T<: AbstractString} = countngrams(x, n, 256, m)
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
struct NGramMatrix{T} <: AbstractMatrix{T}
    s::Vector{T}
    n::Int
    b::Int
    m::Int
end

Base.show(io::IO, n::NGramMatrix) = (print(io, "NGramMatrix($(n.b), $(n.m))"); show(io, n.s))
Base.show(io::IO, ::MIME{Symbol("text/plain")}, n::NGramMatrix) = Base.show(io, n)

NGramIterator(a::NGramMatrix{T}, i) where {T<:AbstractString} = NGramIterator(codeunits(a.s[i]), a.n, a.b)
NGramIterator(a::NGramMatrix{T}, i) where {T<:Vector{U}} where {U<:Integer} = NGramIterator(a.s[i], a.n, a.b)

Base.length(a::NGramMatrix) = length(a.s)
Base.size(a::NGramMatrix) = (a.m, length(a.s))
Base.size(a::NGramMatrix, d) = (d == 1) ? a.m : length(a.s)
subset(a::NGramMatrix{T}, i) where {T<:AbstractString} = NGramMatrix(a.s[i], a.n, a.b, a.m)
Base.getindex(a::NGramMatrix{T}, i) where {T<:AbstractString} = NGramMatrix(a.s[i], a.n, a.b, a.m)
Base.reduce(::typeof(catobs), a::Vector{S}) where {S<:NGramMatrix} = _catobs(a[:])
Base.reduce(::typeof(catobs), a::Matrix{S}) where {S<:NGramMatrix} = _catobs(a[:])
Base.reduce(::typeof(hcat), a::Vector{S}) where {S<:NGramMatrix} = _catobs(a[:])
Base.reduce(::typeof(hcat), a::Matrix{S}) where {S<:NGramMatrix} = _catobs(a[:])
Base.hcat(a::NGramMatrix...) = reduce(catobs, collect(a))
catobs(a::NGramMatrix...) = _catobs(collect(a))
_lastcat(a::Array{S}) where {S<:NGramMatrix} = _catobs(a)
_catobs(a::AbstractVecOrTuple{NGramMatrix}) = NGramMatrix(reduce(vcat, [i.s for i in a]), a[1].n, a[1].b, a[1].m)

Base.Matrix(x::NGramMatrix) = Matrix(SparseMatrixCSC(x))

SparseArrays.SparseMatrixCSC(x::NGramMatrix) = SparseArrays.SparseMatrixCSC{Float32, UInt}(x)

function SparseArrays.SparseMatrixCSC{Tv, Ti}(x::NGramMatrix) where {Tv, Ti <: Integer}
    l = sum(map(i -> length(NGramIterator(x, i)), 1:length(x)))
    I = zeros(Ti, l)
    J = zeros(Ti, l)
    V = ones(Tv, l)
    vid = 1
    for j in 1:length(x)
        for i in NGramIterator(x, j)
            I[vid] = mod(i, x.m) + 1
            J[vid] = j
            vid += 1
        end
    end
    sparse(I, J, V, size(x,1), size(x,2))
end

function mulkernel!(C, A, jB, mA, nA, idxs)
    for iB in idxs
        miB = mod(iB, nA) + 1
        @inbounds for iA in 1:mA
            C[iA, jB] += A[iA, miB]
        end
    end
end

*(A::Matrix, B::NGramMatrix) = mul(A, B)
function mul(A::AbstractMatrix, B::NGramMatrix)
    mA, nA = size(A)
    @assert nA == size(B,1)
    nB = length(B)
    C = zeros(eltype(A), mA, nB)
    for jB in 1:length(B)
        mulkernel!(C, A, jB, mA, nA, NGramIterator(B, jB))
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

function multrans(A::AbstractMatrix, B::NGramMatrix)
    mA, nA = size(A)
    mB = length(B)
    C = zeros(eltype(A), mA, B.m)
    for jB in 1:length(B)
        multkernel!(C, A, jB, mA, B.m, NGramIterator(B, jB))
    end
    return C
end

Zygote.@adjoint function mul(a::AbstractMatrix, b::NGramMatrix)
    return mul(a,b) , Δ -> (multrans(Δ, b),nothing)
end

Zygote.@adjoint function *(a::AbstractMatrix, b::NGramMatrix)
    return mul(a,b) , Δ -> (multrans(Δ, b),nothing)
end
