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

_next_ngram(z, i, it::NGramIterator) = _next_ngram(z, i, it.s, it.n, it.b)
function _next_ngram(z, i, s::Code, n, b)
    z *= b
    z += i ≤ length(s) ? s[i] : string_end_code()
    z -= b^n * (i > n ? s[i - n] : string_start_code())
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
    it1.n === it2.n && it1.b === it2.b && it1.m === it2.m

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

    function NGramMatrix(S::AbstractVector{T}, n::Int=3, b::Int=256, m::Int=2053) where T <: Sequence
        new{T, typeof(S), Int}(S, n, b, m)
    end

    function NGramMatrix(S::AbstractVector{Missing}, n::Int=3, b::Int=256, m::Int=2053)
        new{Missing, typeof(S), Missing}(S, n, b, m)
    end

    function NGramMatrix(S::AbstractVector{T}, n::Int=3, b::Int=256, m::Int=2053) where T <: Maybe{Sequence}
        new{T, typeof(S), Maybe{Int}}(S, n, b, m)
    end

    function NGramMatrix{T, U, V}(S, n::Int=3, b::Int=256, m::Int=2053) where
            {T <: Maybe{Sequence}, U <: AbstractVector{T}, V <: Maybe{Int}}
        new{T, U, V}(convert(U, S), n, b, m)
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

NGramIterator(A::NGramMatrix, i::Integer) = NGramIterator(A.S[i], A.n, A.b, A.m)

Base.length(A::NGramMatrix) = A.m * length(A.S)
Base.size(A::NGramMatrix) = (A.m, length(A.S))
Base.size(A::NGramMatrix, d) = (d == 1) ? A.m : length(A.S)

Base.getindex(X::NGramMatrix, idcs...) = (@boundscheck checkbounds(X, idcs...); _getindex(X, idcs...))
_getindex(X::NGramMatrix, ::Colon, i::Integer) = NGramMatrix(X.S[[i]], X.n, X.b, X.m)
_getindex(X::NGramMatrix, ::Colon, i::AbstractArray) = NGramMatrix(X.S[i], X.n, X.b, X.m)
_getindex(X::NGramMatrix, ::Colon, ::Colon) = NGramMatrix(X.S[:], X.n, X.b, X.m)

subset(a::NGramMatrix, i) = NGramMatrix(a.S[i], a.n, a.b, a.m)

function Base.convert(::Type{<:NGramMatrix{T, U}}, A::NGramMatrix) where {T, U <: AbstractVector{T}}
    NGramMatrix(convert(U, A.S), A.n, A.b, A.m)
end

function Base.promote_rule(::Type{NGramMatrix{T, U, V}}, ::Type{NGramMatrix{A, B, C}}) where
    {T, A, U, B, V, C}
    NGramMatrix{promote_type(T, A), promote_type(U, B), promote_type(V, C)}
end
function Base.promote_rule(::Type{<:NGramMatrix{Missing, U, V}}, ::Type{<:NGramMatrix{A, B, C}}) where
    {A <: Sequence, U, B, V, C}
    X = Union{Missing, A}
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
_mul(A::AbstractMatrix, B::NGramMatrix{T}) where T <: Maybe{Sequence} = _mul(A, B.S, B.n, B.b, B.m)

function _mul(A::AbstractMatrix, S::AbstractVector{T}, n, b, m) where T <: Maybe{Sequence}
    T_res = Missing <: T ? Union{eltype(A), Missing} : eltype(A)
    C = zeros(T_res, size(A, 1), length(S))
    z = _init_z(n, b)
    for (k, s) in enumerate(S)
        _mul_vec!(C, k, A, z, s, n, b, m)
    end
    C
end

Zygote.@adjoint function _mul(A::AbstractMatrix, S::AbstractVector{<:Maybe{Sequence}}, n, b, m)
    return _mul(A, S, n, b, m), Δ -> (_mul_∇A(Δ, A, S, n, b, m), nothing, nothing, nothing, nothing)
end

function _mul_∇A(Δ, A, S, n, b, m)
    ∇A = zero(A)
    z = _init_z(n, b)
    for (k, s) in enumerate(S)
        _∇A_mul_vec!(Δ, k, ∇A, z, s, n, b, m)
    end
    ∇A
end
function ChainRulesCore.rrule(::typeof(_mul_∇A), Δ, A, S, n, b, m)
    y = _mul_∇A(Δ, A, S, n, b, m)
    function _mul_∇A_pullback(Δ₂)
        return (NO_FIELDS, _mul_∇₂A(Δ₂, Δ, A, S, n, b, m)...)
    end
    y, _mul_∇A_pullback
end

function _mul_∇₂A(Δ₂, Δ, A, S, n, b, m)
    _mul(Δ₂, S, n, b, m), nothing, nothing, nothing, nothing, nothing
end

_mul_vec!(C, k, A, z, s::AbstractString, args...) = _mul_vec!(C, k, A, z, codeunits(s), args...)
function _mul_vec!(C, k, A, z, s, n, b, m, ψ=nothing)
    for l in 1:_len(s, n)
        z = _next_ngram(z, l, s, n, b)
        zm = z%m + 1
        for i in 1:size(C, 1)
            @inbounds C[i, k] += A[i, zm]
        end
    end
end
_mul_vec!(C, k, A, z, s::Missing, n, b, m, ψ=missing) = @inbounds C[:, k] .= ψ

_∇A_mul_vec!(Δ, k, ∇A, z, s::AbstractString, args...) = _∇A_mul_vec!(Δ, k, ∇A, z, codeunits(s), args...)
function _∇A_mul_vec!(Δ, k, ∇A, z, s, n, b, m)
    for l in 1:_len(s, n)
        z = _next_ngram(z, l, s, n, b)
        zm = z%m + 1
        for i in 1:size(∇A, 1)
            @inbounds ∇A[i, zm] += Δ[i, k]
        end
    end
end
_∇A_mul_vec!(Δ, k, ∇A, z, s::Missing, n, b, m) = return

Base.hash(M::NGramMatrix, h::UInt) = hash((M.S, M.n, M.b, M.m), h)
(M1::NGramMatrix == M2::NGramMatrix) = isequal(M1.S == M2.S, true) && M1.n == M2.n && M1.b == M2.b && M1.m == M2.m
Base.isequal(M1::NGramMatrix, M2::NGramMatrix) = isequal(M1.S, M2.S) && M1.n == M2.n && M1.b == M2.b && M1.m == M2.m
