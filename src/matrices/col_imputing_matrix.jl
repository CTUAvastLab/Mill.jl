struct ColImputingMatrix{T <: Number, C <: AbstractVector{T}, U <: AbstractMatrix{T}} <: AbstractMatrix{T}
    W::U
    ψ::C
end

Flux.@functor ColImputingMatrix

ColImputingMatrix(W::AbstractMatrix{T}) where T = ColImputingMatrix(W, zeros(T, size(W, 1)))

Flux.@forward ColImputingMatrix.W Base.size, Base.length, Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex

Base.vcat(As::ColImputingMatrix...) = ColImputingMatrix(vcat((A.W for A in As)...), vcat((A.ψ for A in As)...))
function Base.hcat(As::ColImputingMatrix...)
    ArgumentError("It doesn't make sense to hcat ColImputingMatrices") |> throw
end

A::ColImputingMatrix * B::MaybeHotMatrix = _mul(A, B)
Zygote.@adjoint A::ColImputingMatrix * B::MaybeHotMatrix = Zygote.pullback(_mul, A, B)
A::ColImputingMatrix * B::MaybeHotVector = _mul(A, B)
Zygote.@adjoint A::ColImputingMatrix * B::MaybeHotVector = Zygote.pullback(_mul, A, B)

_mul(A::ColImputingMatrix, B::AbstractVecOrMat) = A.W * B
_mul(A::ColImputingMatrix, b::MaybeHotVector{Missing}) = A.ψ
_mul(A::ColImputingMatrix, b::MaybeHotVector{<:Integer}) = A.W * b
_mul(A::ColImputingMatrix, B::MaybeHotMatrix{Missing}) = repeat(A.ψ, 1, size(B, 2))
_mul(A::ColImputingMatrix, B::MaybeHotMatrix{<:Integer}) = A.W * B
_mul(A::ColImputingMatrix, B::MaybeHotMatrix{Maybe{T}}) where {T <: Integer} =
    _impute_maybe_hot(A.W, A.ψ, B.I)

_impute_maybe_hot(W, ψ, I) = _fill_col_mask(W, ψ, I)[1]
function rrule(::typeof(_impute_maybe_hot), W, ψ, I)
    @show "rrule"
    C, m = _fill_col_mask(W, ψ, I)
    function dW_thunk(Δ)
        dW = zero(W)
        for i in skipmissing(I)
            dW[:, i] .+= Δ[:, i]
        end
        dW
    end
    C, Δ -> (NO_FIELDS, @thunk(dW_thunk(Δ)), @thunk(sum(Δ[:, .!m], dims=2)), DoesNotExist())
end
function _fill_col_mask(W, ψ, I)
    m = .!ismissing.(I)
    C = similar(ψ, size(W, 1), length(I))
    C .= ψ
    C[:, m] .= W[:, I[m]]
    C, m
end
