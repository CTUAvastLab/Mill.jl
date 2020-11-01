struct RowImputingMatrix{T <: Number, R <: AbstractVector{T}, U <: AbstractMatrix{T}} <: AbstractMatrix{T}
    W::U
    ψ::R
end

Flux.@functor RowImputingMatrix

RowImputingMatrix(W::AbstractMatrix{T}) where T = RowImputingMatrix(W, zeros(T, size(W, 2)))

Flux.@forward RowImputingMatrix.W Base.size, Base.length, Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex

Base.hcat(As::RowImputingMatrix...) = RowImputingMatrix(hcat((A.W for A in As)...), vcat((A.ψ for A in As)...))
function Base.vcat(As::RowImputingMatrix...)
    ArgumentError("It doesn't make sense to vcat RowImputingMatrices") |> throw
end

A::RowImputingMatrix * b::AbstractVector = (_check_mul(A, b); _mul(A, b))
Zygote.@adjoint A::RowImputingMatrix * b::AbstractVector = (_check_mul(A, b); Zygote.pullback(_mul, A, b))
A::RowImputingMatrix * B::AbstractMatrix = (_check_mul(A, B); _mul(A, B))
Zygote.@adjoint A::RowImputingMatrix * B::AbstractMatrix = (_check_mul(A, B); Zygote.pullback(_mul, A, B))

_mul(A::RowImputingMatrix, B::AbstractVecOrMat) = A.W * B
_mul(A::RowImputingMatrix, ::AbstractVector{Missing}) = A.W * A.ψ
_mul(A::RowImputingMatrix, B::AbstractMatrix{Missing}) = repeat(A.W * A.ψ, 1, size(B, 2))
_mul(A::RowImputingMatrix, B::AbstractVecOrMat{Maybe{T}}) where {T <: Number} = A.W * _impute_row(A.ψ, B)

_impute_row(ψ, B) = _fill_row_mask(ψ, B)[1]
function rrule(::typeof(_impute_row), ψ, B)
    X, m = _fill_row_mask(ψ, B)
    X, Δ -> (NO_FIELDS, @thunk(sum(.!m .* Δ, dims=2)), @thunk(m .* Δ))
end
function _fill_row_mask(ψ, B)
    m = .!ismissing.(B)
    X = similar(ψ, size(B))
    X .= ψ
    X[m] = B[m]
    X, m
end
