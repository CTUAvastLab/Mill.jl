"""
    PostImputingMatrix{T <: Number, U <: AbstractMatrix{T}, V <: AbstractVector{T}} <: AbstractMatrix{T}

A parametrized matrix that fills in a default vector of parameters whenever a \"missing\" column
is encountered during multiplication.

Supports multiplication with [`NGramMatrix`](@ref), [`MaybeHotMatrix`](@ref) and [`MaybeHotVector`](@ref).
For any other `AbstractMatrix` it falls back to standard multiplication.

# Examples
```jlddoctest
julia> A = PostImputingMatrix(ones(2, 2), -ones(2))
2×2 PostImputingMatrix{Float64,Array{Float64,2},Array{Float64,1}}:
W:
 1.0  1.0
 1.0  1.0

ψ:
 -1.0
 -1.0

julia> A * maybehotbatch([1, missing], 1:2)
2×2 Array{Float64,2}:
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
```jlddoctest
julia> PostImputingMatrix([1 2; 3 4])
2×2 PostImputingMatrix{Int64,Array{Int64,2},Array{Int64,1}}:
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
Zygote.@adjoint A::PostImputingMatrix * b::AbstractVector = (_check_mul(A, b); Zygote.pullback(_mul, A, b))
A::PostImputingMatrix * b::MaybeHotVector = (_check_mul(A, b); _mul(A, b))
Zygote.@adjoint A::PostImputingMatrix * b::MaybeHotVector = (_check_mul(A, b); Zygote.pullback(_mul, A, b))
A::PostImputingMatrix * B::AbstractMatrix = (_check_mul(A, B); _mul(A, B))
Zygote.@adjoint A::PostImputingMatrix * B::AbstractMatrix = (_check_mul(A, B); Zygote.pullback(_mul, A, B))
A::PostImputingMatrix * B::MaybeHotMatrix = (_check_mul(A, B); _mul(A, B))
Zygote.@adjoint A::PostImputingMatrix * B::MaybeHotMatrix = (_check_mul(A, B); Zygote.pullback(_mul, A, B))
A::PostImputingMatrix * B::NGramMatrix = (_check_mul(A, B); _mul(A, B))
Zygote.@adjoint A::PostImputingMatrix * B::NGramMatrix = (_check_mul(A, B); Zygote.pullback(_mul, A, B))

_mul(A::PostImputingMatrix, B::AbstractVecOrMat) = A.W * B
_mul(A::PostImputingMatrix, b::MaybeHotVector{Missing}) = A.ψ
_mul(A::PostImputingMatrix, b::MaybeHotVector{<:Integer}) = A.W * b
_mul(A::PostImputingMatrix, B::MaybeHotMatrix{Missing}) = repeat(A.ψ, 1, size(B, 2))
_mul(A::PostImputingMatrix, B::MaybeHotMatrix{<:Integer}) = A.W * B
_mul(A::PostImputingMatrix, B::MaybeHotMatrix{Maybe{T}}) where {T <: Integer} =
    _mul_maybe_hot(A.W, A.ψ, B.I)
_mul(A::PostImputingMatrix, B::NGramMatrix{<:Sequence}) = A.W * B
_mul(A::PostImputingMatrix, B::NGramMatrix{Missing}) = repeat(A.ψ, 1, size(B, 2))
_mul(A::PostImputingMatrix, B::NGramMatrix{Maybe{T}}) where {T <: Sequence} =
    _mul_ngram(A.W, A.ψ, B.s, B.n, B.b, B.m)

_mul_maybe_hot(W, ψ, I) = _impute_maybe_hot(W, ψ, I)[1]
function ChainRulesCore.rrule(::typeof(_mul_maybe_hot), W, ψ, I)
    C, m = _impute_maybe_hot(W, ψ, I)
    function ∇W(Δ)
        dW = zero(W)
        @inbounds for (k, i) in enumerate(I)
            if !ismissing(i)
                @views dW[:, i] .+= Δ[:, k]
            end
        end
        dW
    end
    C, Δ -> (NO_FIELDS, @thunk(∇W(Δ)), @thunk(vec(sum(view(Δ, :, m), dims=2))), DoesNotExist())
end
function _impute_maybe_hot(W, ψ, I)
    m = trues(length(I))
    C = similar(ψ, size(W, 1), length(I))
    C .= ψ
    @inbounds for (k, i) in enumerate(I)
        if !ismissing(I[k])
            m[k] = false
            C[:, k] .= @view W[:, i]
        end
    end
    C, m
end

# TODO rewrite this to less parameters once Zygote allows for composite grads
_mul_ngram(W, ψ, S, n, b, m) = _impute_ngram(W, ψ, S, n, b, m)
function ChainRulesCore.rrule(::typeof(_mul_ngram), W, ψ, S, n, b, m)
    function ∇W(Δ)
        dW = zero(W)
        z = _init_z(n, b)
        for (k, s) in enumerate(S)
            _∇A_mul_vec!(Δ, k, dW, z, s, n, b, m)
        end
        dW
    end
    _impute_ngram(W, ψ, S, n, b, m), Δ -> (NO_FIELDS, @thunk(∇W(Δ)),
                                           @thunk(vec(sum(view(Δ, :, ismissing.(S)), dims=2))),
                                  fill(DoesNotExist(), 4)...)
end

function _impute_ngram(W, ψ, S, n, b, m)
    C = zeros(eltype(ψ), size(W, 1), length(S))
    z = _init_z(n, b)
    for (k, s) in enumerate(S)
        _mul_vec!(C, k, W, z, s, n, b, m, ψ)
    end
    C
end
