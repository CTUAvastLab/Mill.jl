import Base.*
# struct NGramIterator{T} where {T<:Union{Base.CodeUnits{UInt8,S} where S,Vector{I} where I<:Integer}}
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
    idx = (i>n) ? mod(idx,b^n) : idx 
    return(idx, (idx, i + 1))
  elseif i < length(it.s) + n
    idx = mod(idx,b^(n - (i - length(it.s))))
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
    s :: Vector{T}
    n :: Int 
    b :: Int 
    m :: Int
  end

  Represents strings stored in array `s` as ngrams of cardinality `n`. Strings are internally stored as strings and the multiplication with 
  dense matrix is overloaded and `b` is a base for calculation of trigrams. Finally `m` is the modulo applied on indexes of ngrams.

  The structure essentially represents module one-hot representation of strings, where each columns contains one observation (string). 
  Therefore the structure can be viewed as a matrix with `m` rows and `length(s)` columns
"""
struct NGramMatrix{T} <: AbstractMatrix{T}
  s :: Vector{T}
  n :: Int 
  b :: Int 
  m :: Int
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
Base.cat(a::NGramMatrix...) = _catobs(collect(a))
_lastcat(a::Array{S}) where {S<:NGramMatrix} = _catobs(a)
_catobs(a::AbstractVecOrTuple{NGramMatrix}) = NGramMatrix(reduce(vcat, [i.s for i in a]), a[1].n, a[1].b, a[1].m)


function SparseArrays.SparseMatrixCSC(x::Mill.NGramMatrix)
  xx = map(1:length(x)) do  i
    t = zeros(Int, size(x,1),1)
    foreach(j -> t[mod(j, x.m) + 1] += 1,Mill.NGramIterator(x, i))
    sparse(t)
  end
  reduce(hcat, xx)
end

function mulkernel!(C, A, jB, mA, nA, idxs)
  for iB in idxs
    miB = mod(iB, nA) + 1
     @inbounds for iA in 1:mA
        C[iA, jB] += A[iA, miB]
    end
  end
end

function mul(A::Matrix, B::NGramMatrix)
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

function multrans(A::Matrix, B::NGramMatrix)
  mA, nA = size(A)
  mB = length(B)
  C = zeros(eltype(A), mA, B.m)
  for jB in 1:length(B)
    multkernel!(C, A, jB, mA, B.m, NGramIterator(B, jB))
  end
  return C
end

a::Flux.Tracker.TrackedMatrix * b::NGramMatrix = Flux.Tracker.track(mul, a, b)
a::Matrix * b::NGramMatrix = mul(a, b)
Flux.Tracker.@grad function mul(a::Flux.Tracker.TrackedMatrix, b::NGramMatrix)
  return mul(Flux.data(a),b) , Δ -> (multrans(Δ, b),nothing)
end
