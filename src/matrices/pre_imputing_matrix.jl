struct PreImputingMatrix{T <: Number, R <: AbstractVector{T}, U <: AbstractMatrix{T}} <: AbstractMatrix{T}
    W::U
    ψ::R
end

Flux.@functor PreImputingMatrix

PreImputingMatrix(W::AbstractMatrix{T}) where T = PreImputingMatrix(W, zeros(T, size(W, 2)))

Flux.@forward PreImputingMatrix.W Base.size, Base.length, Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex

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
function rrule(::typeof(_mul_maybe), ψ, B)
    X, m = _impute_row(ψ, B)
    X, Δ -> (NO_FIELDS, @thunk(sum(.!m .* Δ, dims=2)), @thunk(m .* Δ))
end
function _impute_row(ψ, B)
    m = .!ismissing.(B)
    X = similar(ψ, size(B))
    X .= ψ
    X[m] = @view B[m]
    X, m
end
