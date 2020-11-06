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

A::ColImputingMatrix * b::AbstractVector = (_check_mul(A, b); _mul(A, b))
Zygote.@adjoint A::ColImputingMatrix * b::AbstractVector = (_check_mul(A, b); Zygote.pullback(_mul, A, b))
A::ColImputingMatrix * B::AbstractMatrix = (_check_mul(A, B); _mul(A, B))
Zygote.@adjoint A::ColImputingMatrix * B::AbstractMatrix = (_check_mul(A, B); Zygote.pullback(_mul, A, B))
A::ColImputingMatrix * b::MaybeHotVector = (_check_mul(A, b); _mul(A, b))
Zygote.@adjoint A::ColImputingMatrix * b::MaybeHotVector = (_check_mul(A, b); Zygote.pullback(_mul, A, b))
A::ColImputingMatrix * B::MaybeHotMatrix = (_check_mul(A, B); _mul(A, B))
Zygote.@adjoint A::ColImputingMatrix * B::MaybeHotMatrix = (_check_mul(A, B); Zygote.pullback(_mul, A, B))
A::ColImputingMatrix * B::NGramMatrix = (_check_mul(A, B); _mul(A, B))
Zygote.@adjoint A::ColImputingMatrix * B::NGramMatrix = (_check_mul(A, B); Zygote.pullback(_mul, A, B))

_mul(A::ColImputingMatrix, B::AbstractVecOrMat) = A.W * B
_mul(A::ColImputingMatrix, b::MaybeHotVector{Missing}) = A.ψ
_mul(A::ColImputingMatrix, b::MaybeHotVector{<:Integer}) = A.W * b
_mul(A::ColImputingMatrix, B::MaybeHotMatrix{Missing}) = repeat(A.ψ, 1, size(B, 2))
_mul(A::ColImputingMatrix, B::MaybeHotMatrix{<:Integer}) = A.W * B
_mul(A::ColImputingMatrix, B::MaybeHotMatrix{Maybe{T}}) where {T <: Integer} =
    _mul_maybe_hot(A.W, A.ψ, B.I)
_mul(A::ColImputingMatrix, B::NGramMatrix{<:Sequence}) = A.W * B
_mul(A::ColImputingMatrix, B::NGramMatrix{Missing}) = repeat(A.ψ, 1, size(B, 2))
_mul(A::ColImputingMatrix, B::NGramMatrix{Maybe{T}}) where {T <: Sequence} =
    _mul_ngram(A.W, A.ψ, B.s, B.n, B.b, B.m)

_mul_maybe_hot(W, ψ, I) = _impute_maybe_hot(W, ψ, I)[1]
function rrule(::typeof(_mul_maybe_hot), W, ψ, its)
    C, m = _impute_maybe_hot(W, ψ, I)
    function dW_thunk(Δ)
        dW = zero(W)
        for (k, i) in enumerate(I)
            if !ismissing(i)
                @views dW[:, i] .+= Δ[:, k]
            end
        end
        dW
    end
    C, Δ -> (NO_FIELDS, @thunk(dW_thunk(Δ)), @thunk(sum(view(Δ, :, .!m), dims=2)), DoesNotExist())
end
function _impute_maybe_hot(W, ψ, I)
    m = .!ismissing.(I)
    C = similar(ψ, size(W, 1), length(I))
    C .= ψ
    @views C[:, m] .= W[:, I[m]]
    C, m
end

# TODO rewrite this to less parameters once Zygote allows for composite grads
_mul_ngram(W, ψ, S, n, b, m) = _impute_ngram(W, ψ, S, n, b, m)
function rrule(::typeof(_mul_ngram), W, ψ, S, n, b, m)
    function dW_thunk(Δ)
        dW = zero(W)
        for (k, s) in enumerate(S)
            _dA_mul_vec!(view(Δ, :, k), dW, NGramIterator(s, n, b, m))
        end
        dW
    end
    _impute_ngram(W, ψ, S, n, b, m), Δ -> (NO_FIELDS, @thunk(dW_thunk(Δ)),
                                  @thunk(sum(view(Δ, :, ismissing.(S)), dims=2)),
                                  fill(DoesNotExist(), 4)...)
end

function _impute_ngram(W, ψ, S, n, b, m)
    C = similar(ψ, size(W, 1), length(S))
    for (k, s) in enumerate(S)
        _mul_vec!(view(C, :, k), W, NGramIterator(s, n, b, m), ψ)
    end
    C
end
