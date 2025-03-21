const Code = Union{AbstractVector{<:Integer}, CodeUnits}
const Sequence = Union{AbstractString, Code}

"""
    NGramIterator{T}

Iterates over ngram codes of collection of integers `s` using [`Mill.string_start_code()`](@ref) and [`Mill.string_end_code()`](@ref) for padding. NGram codes are computed as in positional number systems, where items of `s` are digits, `b` is the base, and `m` is modulo.

In order to reduce collisions when mixing ngrams of different order one should avoid zeros and negative integers in `s` and should set base `b` to the expected number of unique tokens in `s`.

See also: [`NGramMatrix`](@ref), [`ngrams`](@ref), [`ngrams!`](@ref), [`countngrams`](@ref),
    [`countngrams!`](@ref).
"""
struct NGramIterator{T}
    s::T
    n::Int
    b::Int
    m::Int

    NGramIterator(s::T, n::Int=3, b::Int=256, m::Int=typemax(Int)) where T <: Maybe{Code} = new{T}(s, n, b, m)
end

"""
    NGramIterator(s, n=3, b=256, m=typemax(Int))

Construct an [`NGramIterator`](@ref). If `s` is an `AbstractString` it is first converted
to integers with `Base.codeunits`.

# Examples
```jldoctest
julia> NGramIterator("deadbeef", 3, 256, 17) |> collect
10-element Vector{Int64}:
  2
 16
  9
  9
  6
 10
 11
 15
  2
  6

julia> NGramIterator(collect(1:9), 3, 10, 1009) |> collect
11-element Vector{Int64}:
 221
 212
 123
 234
 345
 456
 567
 678
 789
 893
 933

julia> Mill.string_start_code()
0x02

julia> Mill.string_end_code()
0x03
```

See also: [`NGramMatrix`](@ref), [`ngrams`](@ref), [`ngrams!`](@ref), [`countngrams`](@ref),
    [`countngrams!`](@ref).
"""
NGramIterator(s::AbstractString, args...) = NGramIterator(codeunits(s), args...)

Base.eltype(::NGramIterator) = Int

@inline _len(s::Sequence, n) = length(s) + n - 1
Base.length(it::NGramIterator) = _len(it.s, it.n)
Base.length(it::NGramIterator{Missing}) = 0

_next_ngram(z, i, it::NGramIterator) = _next_ngram(z, i, it.s, it.n, it.b, it.b^it.n)
function _next_ngram(z, i, s::Code, n, b, bn)
    z *= b
    z += i ≤ length(s) ? s[i] : string_end_code()
    z -= bn * (i > n ? s[i - n] : string_start_code())
end

Base.iterate(it::NGramIterator{Missing}) = nothing
function Base.iterate(it::NGramIterator, (z, i) = (_init_z(it), 1))
    if i ≤ length(it)
        z = _next_ngram(z, i, it)
        z % it.m, (z, i+1)
    end
end

_init_z(it) = _init_z(it.n, it.b)
function _init_z(n::T, b::T) where T
    z = zero(T)
    for _ in 1:n
        z *= b
        z += one(T)
    end
    string_start_code() * z
end

Base.hash(it::NGramIterator{T}, h::UInt) where {T} = hash((string(T), it.s, it.n, it.b, it.m), h)
(it1::NGramIterator{T} == it2::NGramIterator{T}) where {T} = it1.s == it2.s &&
    it1.n ≡ it2.n && it1.b ≡ it2.b && it1.m ≡ it2.m

"""
    ngrams!(o, x, n=3, b=256)

Store codes of `n` grams of `x` using base `b` to `o`.

# Examples
```jldoctest
julia> o = zeros(Int, 5)
5-element Vector{Int64}:
 0
 0
 0
 0
 0

julia> ngrams!(o, "foo", 3, 256)
5-element Vector{Int64}:
  131686
  157295
 6713199
 7302915
 7275267
```

See also: [`ngrams`](@ref), [`countngrams`](@ref),
    [`countngrams!`](@ref), [`NGramMatrix`](@ref), [`NGramIterator`](@ref).
"""
function ngrams!(o, x::Sequence, args...)
    for (i, idx) in enumerate(NGramIterator(x, args...))
        o[i] = idx
    end
    o
end

"""
    ngrams(o, x, n=3, b=256)

Return codes of `n` grams of `x` using base `b`.

# Examples
```jldoctest
julia> ngrams("foo", 3, 256)
5-element Vector{Int64}:
  131686
  157295
 6713199
 7302915
 7275267
```

See also: [`ngrams!`](@ref), [`countngrams`](@ref),
    [`countngrams!`](@ref), [`NGramMatrix`](@ref), [`NGramIterator`](@ref).
"""
ngrams(x::Sequence, args...) = collect(NGramIterator(x, args...))

"""
    countngrams!(o, x, n, b, m=length(o))

Count the number of of `n` grams of `x` using base `b` and modulo `m` and store the result to `o`.

# Examples
```jldoctest
julia> o = zeros(Int, 5)
5-element Vector{Int64}:
 0
 0
 0
 0
 0

julia> countngrams!(o, "foo", 3, 256)
5-element Vector{Int64}:
 2
 1
 1
 0
 1
```

See also: [`countngrams`](@ref), [`ngrams`](@ref), [`ngrams!`](@ref),
    [`NGramMatrix`](@ref), [`NGramIterator`](@ref).
"""
countngrams!(o, x::Sequence, n::Int, b::Int) = countngrams!(o, x, n, b, length(o))
function countngrams!(o, x::Sequence, args...)
    for idx in NGramIterator(x, args...)
        o[idx + 1] += 1
    end
    o
end

"""
    countngrams(o, x, n, b, m)

Count the number of of `n` grams of `x` using base `b` and modulo `m` into a vector of length `m`
in case `x` is a single sequence or into a matrix with `m` rows if `x` is an iterable of sequences.

# Examples
```jldoctest
julia> countngrams("foo", 3, 256, 5)
5-element Vector{Int64}:
 2
 1
 1
 0
 1

julia> countngrams(["foo", "bar"], 3, 256, 5)
5×2 Matrix{Int64}:
 2  1
 1  0
 1  2
 0  0
 1  2
```

See also: [`countngrams!`](@ref), [`ngrams`](@ref), [`ngrams!`](@ref),
    [`NGramMatrix`](@ref), [`NGramIterator`](@ref).
"""
countngrams(x, n, b, m) = countngrams!(zeros(Int, m), x, n, b)
function countngrams(x::AbstractVector{<:Sequence}, n, b, m)
    o = zeros(Int, m, length(x))
    for (i, s) in enumerate(x)
        countngrams!(view(o, :, i), x[i], n, b, m)
    end
    o
end

"""
    NGramMatrix{T, U, V} <: AbstractMatrix{U}

A matrix-like structure for lazily representing sequences like strings as ngrams of
cardinality `n` using `b` as a base for calculations and `m` as the modulo. Therefore, the matrix
has `m` rows and one column for representing each sequence. Missing sequences are supported.

See also: [`NGramIterator`](@ref), [`ngrams`](@ref), [`ngrams!`](@ref), [`countngrams`](@ref),
    [`countngrams!`](@ref).
"""
struct NGramMatrix{T, U, V} <: AbstractMatrix{V}
    S::U
    n::Int
    b::Int
    m::Int


    function NGramMatrix{T, U, V}(s, n=3, b=256, m=2053) where
            {T <: Maybe{Sequence}, U <: AbstractVector{T}, V <: Maybe{Int}}
        new{T, U, V}(convert(U, s), n, b, m)
    end
end

"""
    NGramMatrix(s, n=3, b=256, m=2053)

Construct an [`NGramMatrix`](@ref). `s` can either be a single sequence or any `AbstractVector`.

# Examples
```jldoctest
julia> NGramMatrix([1,2,3])
2053×1 NGramMatrix{Vector{Int64}, Vector{Vector{Int64}}, Int64}:
 [1, 2, 3]

julia> NGramMatrix(["a", missing, "c"], 2, 128)
2053×3 NGramMatrix{Union{Missing, String}, Vector{Union{Missing, String}}, Union{Missing, Int64}}:
 "a"
 missing
 "c"
```

See also: [`NGramIterator`](@ref), [`ngrams`](@ref), [`ngrams!`](@ref), [`countngrams`](@ref),
    [`countngrams!`](@ref).
"""
NGramMatrix(s::Maybe{Sequence}, args...) = NGramMatrix([s], args...)
function NGramMatrix(S::AbstractVector{T}, args...) where T <: Sequence
    NGramMatrix{T, typeof(S), Int}(S, args...)
end
function NGramMatrix(S::AbstractVector{Missing}, args...)
    NGramMatrix{Missing, typeof(S), Missing}(S, args...)
end
function NGramMatrix(S::AbstractVector{T}, args...) where T <: Maybe{Sequence}
    NGramMatrix{T, typeof(S), Maybe{Int}}(S, args...)
end

NGramIterator(A::NGramMatrix, i::Integer) = NGramIterator(A.S[i], A.n, A.b, A.m)

Base.length(A::NGramMatrix) = A.m * length(A.S)
Base.size(A::NGramMatrix) = (A.m, length(A.S))
Base.size(A::NGramMatrix, d) = (d == 1) ? A.m : length(A.S)

Base.getindex(X::NGramMatrix, idcs...) = (@boundscheck checkbounds(X, idcs...); _getindex(X, idcs...))
_getindex(X::NGramMatrix, ::Colon, i::Integer) = NGramMatrix(X.S[[i]], X.n, X.b, X.m)
_getindex(X::NGramMatrix, ::Colon, i::Union{AbstractArray, UnitRange}) = NGramMatrix(X.S[i], X.n, X.b, X.m)
_getindex(X::NGramMatrix, ::Colon, ::Colon) = NGramMatrix(X.S[:], X.n, X.b, X.m)

function Base.convert(::Type{<:NGramMatrix{T, U}}, A::NGramMatrix) where {T, U <: AbstractVector{T}}
    NGramMatrix(convert(U, A.S), A.n, A.b, A.m)
end

function Base.promote_rule(::Type{NGramMatrix{T, U, V}}, ::Type{NGramMatrix{A, B, C}}) where
    {T, A, U, B, V, C}
    NGramMatrix{promote_type(T, A), promote_type(U, B), promote_type(V, C)}
end
function Base.promote_rule(::Type{<:NGramMatrix{Missing, U, V}}, ::Type{<:NGramMatrix{A, B, C}}) where
    {A <: Sequence, U, B, V, C}
    X = Maybe{A}
    NGramMatrix{X, promote_type(U, B){X}, promote_type(V, C)}
end
Base.promote_rule(t1::Type{<:NGramMatrix{<:Sequence}}, t2::Type{<:NGramMatrix{Missing}}) = promote_rule(t2, t1)

function _check_nbm(As::AbstractVecOrTuple{NGramMatrix})
    n, b, m = As[1].n, As[1].b, As[1].m
    if any(!isequal(n), (A.n for A in As)) ||
        any(!isequal(b), (A.b for A in As)) ||
        any(!isequal(m), (A.m for A in As))
        DimensionMismatch(
                          "Matrices do not have the same n, b, or m."
                         ) |> throw
    end
    n, b, m
end

Base.hcat(As::T...) where T <: NGramMatrix = _typed_hcat(T, As)
Base.hcat(As::NGramMatrix...) = _typed_hcat(_promote_types(As...), As)
function _typed_hcat(::Type{T}, As::Tuple{Vararg{NGramMatrix}}) where T <: NGramMatrix
    T(vcat([A.S for A in As]...), _check_nbm(As)...)
end

Base.reduce(::typeof(hcat), As::Vector{<:NGramMatrix}) = _typed_hcat(mapreduce(typeof, promote_type, As), As)
function _typed_hcat(::Type{T}, As::AbstractVector{<:NGramMatrix}) where T <: NGramMatrix
    T(reduce(vcat, [A.S for A in As]), _check_nbm(As)...)
end

SparseArrays.SparseMatrixCSC(x::NGramMatrix) = SparseArrays.SparseMatrixCSC{Int64, UInt}(x)
function SparseArrays.SparseMatrixCSC{Tv, Ti}(x::NGramMatrix) where {Tv, Ti <: Integer}
    size(x, 2) == 0 && return sparse(Ti[],Ti[],Tv[], size(x,1), size(x,2))
    l = sum(map(i -> length(NGramIterator(x, i)), axes(x, 2)))
    I = zeros(Ti, l)
    J = zeros(Ti, l)
    V = ones(Tv, l)
    vid = 1
    for j in axes(x, 2)
        for i in NGramIterator(x, j)
            I[vid] = i + 1
            J[vid] = j
            vid += 1
        end
    end
    sparse(I, J, V, size(x,1), size(x,2))
end

A::AbstractMatrix * B::NGramMatrix = (_check_mul(A, B); _mul(A, B))
@opt_out rrule(::typeof(*), ::AbstractVecOrMat{<:Union{Real, Complex}}, ::NGramMatrix)

_mul(A::AbstractMatrix, B::NGramMatrix{Missing}) = fill(missing, size(A, 1), size(B, 2))
_mul(A::AbstractMatrix, B::NGramMatrix{T}) where T <: Maybe{Sequence} = _mul_ngram(A, B)

_mul_ngram(A, B) = _mul_ngram_forw(A, B)
function _mul_ngram_forw(A::AbstractMatrix, B::NGramMatrix{T}) where T <: Maybe{Sequence}
    T_res = Missing <: T ? Maybe{eltype(A)} : eltype(A)
    C = zeros(T_res, size(A, 1), length(B.S))
    z = _init_z(B.n, B.b)
    bn = B.b^B.n
    for (k, s) in enumerate(B.S)
        _mul_ngram_vec!(s, A, B, bn, C, k, z)
    end
    C
end

_mul_ngram_vec!(::Missing, A, B, bn, C, k, z, ψ=missing) = @inbounds C[:, k] .= ψ
_mul_ngram_vec!(s::AbstractString, args...) = _mul_ngram_vec!(codeunits(s), args...)
function _mul_ngram_vec!(s, A, B, bn, C, k, z, ψ=nothing)
    for l in 1:_len(s, B.n)
        z = mod(_next_ngram(z, l, s, B.n, B.b, bn), B.m)
        zi = z + 1
        for i in axes(C, 1)
            @inbounds C[i, k] += A[i, zi]
        end
    end
end

function ChainRulesCore.rrule(::typeof(_mul_ngram), A::AbstractMatrix, B::NGramMatrix)
    return _mul_ngram_forw(A, B), Δ -> (NoTangent(), _mul_ngram_∇A(Δ, A, B), NoTangent())
end

function _mul_ngram_∇A(Δ, A, B)
    ∇A = zero(A)
    z = _init_z(B.n, B.b)
    bn = B.b^B.n
    for (k, s) in enumerate(B.S)
        _∇A_mul_ngram_vec!(Δ, s, B, bn, ∇A, k, z)
    end
    ∇A
end

_∇A_mul_ngram_vec!(Δ, s::Missing, args...) = return
_∇A_mul_ngram_vec!(Δ, s::AbstractString, B, bn, ∇A, k, z) = _∇A_mul_ngram_vec!(Δ, codeunits(s), B, bn, ∇A, k, z)
function _∇A_mul_ngram_vec!(Δ, s, B, bn, ∇A, k, z)
    for l in 1:_len(s, B.n)
        z = mod(_next_ngram(z, l, s, B.n, B.b, bn), B.m)
        zi = z + 1
        for i in axes(∇A, 1)
            @inbounds ∇A[i, zi] += Δ[i, k]
        end
    end
end

Base.hash(M::NGramMatrix, h::UInt) = hash((M.S, M.n, M.b, M.m), h)
(M1::NGramMatrix == M2::NGramMatrix) = M1.S == M2.S && M1.n == M2.n && M1.b == M2.b && M1.m == M2.m
Base.isequal(M1::NGramMatrix, M2::NGramMatrix) = isequal(M1.S, M2.S) && M1.n == M2.n && M1.b == M2.b && M1.m == M2.m
