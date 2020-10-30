struct ImputingMatrix{T <: Number, U <: AbstractMatrix{T}, V <: AbstractVector{T}} <: AbstractMatrix{T}
    W::U
    ψ::V
end

Flux.@functor ImputingMatrix

ImputingMatrix(W::AbstractMatrix{T}) where T = ImputingMatrix(W, zeros(T, size(W, 2)))

Flux.@forward ImputingMatrix.W Base.size, Base.length, Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex

Base.vcat(As::ImputingMatrix...) = ImputingMatrix(vcat((A.W for A in As)...), vcat((A.ψ for A in As)...))

function print_array(io::IO, A::ImputingMatrix)
    println(io, "W:")
    print_array(io, A.W)
    println(io, "\nψ:")
    print_array(io, A.ψ)
end

function Flux.params!(p::Params, A::ImputingMatrix, seen=IdSet())
  A in seen && return
  push!(seen, A)
  push!(p, A.W, A.ψ)
end

A::ImputingMatrix * B::AbstractVector = _mul(A, B)
Zygote.@adjoint A::ImputingMatrix * B::AbstractVector = Zygote.pullback(_mul, A, B)
A::ImputingMatrix * B::AbstractMatrix = _mul(A, B)
Zygote.@adjoint A::ImputingMatrix * B::AbstractMatrix = Zygote.pullback(_mul, A, B)

_mul(A::ImputingMatrix, B::AbstractVecOrMat) = A.W * B
_mul(A::ImputingMatrix, ::AbstractVector{Missing}) = A.W * A.ψ
_mul(A::ImputingMatrix, B::AbstractMatrix{Missing}) = repeat(A.W * A.ψ, 1, size(B, 2))
_mul(A::ImputingMatrix, B::AbstractVecOrMat{Maybe{T}}) where T = A.W * _fill_in(A.ψ, B)

_fill_in(ψ, B) = _fill_mask(ψ, B)[1]
function rrule(::typeof(_fill_in), ψ, B)
    X, m = _fill_mask(ψ, B)
    _fill_in_dψ(Δ) = (dψ = deepcopy(Δ); dψ[m] .= 0; sum(dψ, dims=2))
    _fill_in_dB(Δ) = (dB = deepcopy(Δ); dB[.!m] .= 0; dB)
    X, Δ -> (NO_FIELDS, @thunk(_fill_in_dψ(Δ)), @thunk(_fill_in_dB(Δ)))
end

function _fill_mask(ψ::AbstractVector{T}, B) where T
    m = .!ismissing.(B)
    X = similar(B, T)
    X .= ψ
    X[m] = B[m]
    X, m
end

ImputingDense(d::Dense) = Dense(ImputingMatrix(d.W), d.b, d.σ)
ImputingDense(args...) = ImputingDense(Dense(args...))

function Base.show(io::IO, l::Dense{F, <:ImputingMatrix}) where F
  print(io, "ImputingDense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end
