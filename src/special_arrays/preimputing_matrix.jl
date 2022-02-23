"""
    PreImputingMatrix{T <: Number, U <: AbstractMatrix{T}, V <: AbstractVector{T}} <: AbstractMatrix{T}

A parametrized matrix that fills in elements from a default vector of parameters whenever a `missing`
element is encountered during multiplication.

# Examples
```jldoctest
julia> A = PreImputingMatrix(ones(2, 2), -ones(2))
2×2 PreImputingMatrix{Float64, Matrix{Float64}, Vector{Float64}}:
W:
 1.0  1.0
 1.0  1.0

ψ:
 -1.0  -1.0

julia> A * [0 1; missing -1]
2×2 Matrix{Float64}:
 -1.0  0.0
 -1.0  0.0
```

See also: [`PreImputingMatrix`](@ref).
"""
struct PreImputingMatrix{T <: Number, U <: AbstractMatrix{T}, V <: AbstractVector{T}} <: AbstractMatrix{T}
    W::U
    ψ::V
end

Flux.@functor PreImputingMatrix

"""
    PreImputingMatrix(W::AbstractMatrix{T}, ψ=zeros(T, size(W, 2))) where T

Construct a [`PreImputingMatrix`](@ref) with multiplication parameters `W` and default parameters `ψ`.

# Examples
```jldoctest
julia> PreImputingMatrix([1 2; 3 4])
2×2 PreImputingMatrix{Int64, Matrix{Int64}, Vector{Int64}}:
W:
 1  2
 3  4

ψ:
 0  0
```

See also: [`PostImputingMatrix`](@ref).
"""
PreImputingMatrix(W::AbstractMatrix{T}) where T = PreImputingMatrix(W, zeros(T, size(W, 2)))

Flux.@forward PreImputingMatrix.W Base.size, Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex

Base.hcat(As::PreImputingMatrix...) = PreImputingMatrix(hcat((A.W for A in As)...), vcat((A.ψ for A in As)...))
function Base.vcat(As::PreImputingMatrix...)
    ArgumentError("It doesn't make sense to vcat PreImputingMatrices") |> throw
end

A::PreImputingMatrix * b::AbstractVector = (_check_mul(A, b); _mul(A, b))
Zygote.@adjoint A::PreImputingMatrix * b::AbstractVector = (_check_mul(A, b); Zygote.pullback(_mul, A, b))
A::PreImputingMatrix * B::AbstractMatrix = (_check_mul(A, B); _mul(A, B))
Zygote.@adjoint A::PreImputingMatrix * B::AbstractMatrix = (_check_mul(A, B); Zygote.pullback(_mul, A, B))

_mul(A::PreImputingMatrix, B::AbstractVecOrMat) = A.W * B
_mul(A::PreImputingMatrix, ::AbstractVector{Missing}) = A.W * A.ψ
_mul(A::PreImputingMatrix, B::AbstractMatrix{Missing}) = repeat(A.W * A.ψ, 1, size(B, 2))
_mul(A::PreImputingMatrix, B::AbstractVecOrMat{Maybe{T}}) where {T <: Number} = A.W * _mul_maybe(A.ψ, B)

_mul_maybe(ψ, B) = _impute_row(ψ, B)[1]
function ChainRulesCore.rrule(::typeof(_mul_maybe), ψ, B)
    X, m = _impute_row(ψ, B)
    X, Δ -> (NoTangent(), @thunk(vec(sum(.!m .* Δ, dims=2))), @thunk(m .* Δ))
end
function _impute_row(ψ, B)
    m = .!ismissing.(B)
    X = similar(ψ, size(B))
    X .= ψ
    X[m] = @view B[m]
    X, m
end
