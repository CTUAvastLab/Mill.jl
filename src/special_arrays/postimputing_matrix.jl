"""
    PostImputingMatrix{T <: Number, U <: AbstractMatrix{T}, V <: AbstractVector{T}} <: AbstractMatrix{T}

A parametrized matrix that fills in a default vector of parameters whenever a \"missing\" column
is encountered during multiplication.

Supports multiplication with [`NGramMatrix`](@ref), [`MaybeHotMatrix`](@ref) and [`MaybeHotVector`](@ref).
For any other `AbstractMatrix` it falls back to standard multiplication.

# Examples
```jldoctest
julia> A = PostImputingMatrix(ones(2, 2), -ones(2))
2×2 PostImputingMatrix{Float64, Matrix{Float64}, Vector{Float64}}:
W:
 1.0  1.0
 1.0  1.0

ψ:
 -1.0
 -1.0

julia> A * maybehotbatch([1, missing], 1:2)
2×2 Matrix{Float64}:
 1.0  -1.0
 1.0  -1.0
```

See also: [`PreImputingMatrix`](@ref).
"""
struct PostImputingMatrix{T <: Number, U <: AbstractMatrix{T}, V <: AbstractVector{T}} <: AbstractMatrix{T}
    W::U
    ψ::V
end

Flux.@functor PostImputingMatrix

"""
    PostImputingMatrix(W::AbstractMatrix{T}, ψ=zeros(T, size(W, 1))) where T

Construct a [`PostImputingMatrix`](@ref) with multiplication parameters `W` and default parameters `ψ`.

# Examples
```jldoctest
julia> PostImputingMatrix([1 2; 3 4])
2×2 PostImputingMatrix{Int64, Matrix{Int64}, Vector{Int64}}:
W:
 1  2
 3  4

ψ:
 0
 0
```

See also: [`PreImputingMatrix`](@ref).
"""
PostImputingMatrix(W::AbstractMatrix{T}) where T = PostImputingMatrix(W, zeros(T, size(W, 1)))

Flux.@forward PostImputingMatrix.W Base.size, Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex

Base.vcat(As::PostImputingMatrix...) = PostImputingMatrix(vcat((A.W for A in As)...), vcat((A.ψ for A in As)...))
function Base.hcat(As::PostImputingMatrix...)
    ArgumentError("It doesn't make sense to hcat PostImputingMatrices") |> throw
end

A::PostImputingMatrix * b::AbstractVector = (_check_mul(A, b); _mul(A, b))
A::PostImputingMatrix * b::MaybeHotVector = (_check_mul(A, b); _mul(A, b))
A::PostImputingMatrix * B::AbstractMatrix = (_check_mul(A, B); _mul(A, B))
A::PostImputingMatrix * B::MaybeHotMatrix = (_check_mul(A, B); _mul(A, B))
A::PostImputingMatrix * B::NGramMatrix = (_check_mul(A, B); _mul(A, B))
@opt_out rrule(::typeof(*), ::PostImputingMatrix, ::AbstractVecOrMat{<:Union{Real, Complex}})
@opt_out rrule(::typeof(*), ::PostImputingMatrix, ::NGramMatrix)

_mul(A::PostImputingMatrix, B::AbstractVecOrMat) = A.W * B
_mul(A::PostImputingMatrix, b::MaybeHotVector{Missing}) = A.ψ
_mul(A::PostImputingMatrix, b::MaybeHotVector{<:Integer}) = A.W * b
_mul(A::PostImputingMatrix, B::MaybeHotMatrix{Missing}) = repeat(A.ψ, 1, size(B, 2))
_mul(A::PostImputingMatrix, B::MaybeHotMatrix{<:Integer}) = A.W * B
_mul(A::PostImputingMatrix, B::MaybeHotMatrix{Maybe{T}}) where {T <: Integer} = _mul_pi_maybe_hot(A, B)
_mul(A::PostImputingMatrix, B::NGramMatrix{<:Sequence}) = A.W * B
_mul(A::PostImputingMatrix, B::NGramMatrix{Missing}) = repeat(A.ψ, 1, size(B, 2))
_mul(A::PostImputingMatrix, B::NGramMatrix{Maybe{T}}) where {T <: Sequence} = _mul_pi_ngram(A, B)

_mul_pi_maybe_hot(A, B) = _postimpute_maybe_hot(A, B)[1]
function ChainRulesCore.rrule(::typeof(_mul_pi_maybe_hot), A, B)
    C, m = _postimpute_maybe_hot(A, B)
    function ∇W(Δ)
        dW = zero(A.W)
        @inbounds for (k, j) in enumerate(B.I)
            if !ismissing(j)
                for i in 1:size(dW, 1)
                    dW[i, j] += Δ[i, k]
                end
            end
        end
        dW
    end
    C, Δ -> begin
        NoTangent(),
        Tangent{typeof(A)}(W = @thunk(∇W(Δ)),
                           ψ = @thunk(vec(sum(view(Δ, :, m), dims=2)))),
        NoTangent()
    end
end
function _postimpute_maybe_hot(A, B)
    m = trues(length(B.I))
    C = similar(A.ψ, size(A.W, 1), length(B.I))
    C .= A.ψ
    @inbounds for (j, k) in enumerate(B.I)
        if !ismissing(k)
            m[j] = false
            for i in 1:size(C, 1)
                C[i, j] = A.W[i, k]
            end
        end
    end
    C, m
end

_mul_pi_ngram(A, B) = _postimpute_ngram(A, B)
function ChainRulesCore.rrule(::typeof(_mul_pi_ngram), A, B)
    function ∇W(Δ)
        dW = zero(A.W)
        z = _init_z(B.n, B.b)
        for (k, s) in enumerate(B.S)
            _∇A_mul_vec!(Δ, k, dW, z, B, s)
        end
        dW
    end
    _postimpute_ngram(A, B), Δ -> begin
        NoTangent(),
        Tangent{typeof(A)}(W = @thunk(∇W(Δ)),
                           ψ = @thunk(vec(sum(view(Δ, :, ismissing.(B.S)), dims=2)))),
        NoTangent()
    end
end

function _postimpute_ngram(A, B)
    C = zeros(eltype(A.ψ), size(A.W, 1), length(B.S))
    z = _init_z(B.n, B.b)
    for (k, s) in enumerate(B.S)
        _mul_vec!(C, k, A.W, z, B, s, A.ψ)
    end
    C
end
