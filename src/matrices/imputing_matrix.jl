struct RowImputingMatrix{T <: Number, R <: AbstractVector{T}, U <: AbstractMatrix{T}} <: AbstractMatrix{T}
    W::U
    ψ::R
end

struct ColImputingMatrix{T <: Number, C <: AbstractVector{T}, U <: AbstractMatrix{T}} <: AbstractMatrix{T}
    W::U
    ψ::C
end

const ImputingMatrix{T, C, U} = Union{RowImputingMatrix{T, C, U}, ColImputingMatrix{T, C, U}}

Flux.@functor RowImputingMatrix
Flux.@functor ColImputingMatrix

RowImputingMatrix(W::AbstractMatrix{T}) where T = RowImputingMatrix(W, zeros(T, size(W, 2)))
ColImputingMatrix(W::AbstractMatrix{T}) where T = ColImputingMatrix(W, zeros(T, size(W, 1)))

Flux.@forward RowImputingMatrix.W Base.size, Base.length, Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex
Flux.@forward ColImputingMatrix.W Base.size, Base.length, Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex

Base.hcat(As::RowImputingMatrix...) = RowImputingMatrix(hcat((A.W for A in As)...), hcat((A.ψ for A in As)...))
Base.vcat(As::ColImputingMatrix...) = ColImputingMatrix(vcat((A.W for A in As)...), vcat((A.ψ for A in As)...))

_print_params(io::IO, A::RowImputingMatrix) = print_array(io, A.ψ')
_print_params(io::IO, A::ColImputingMatrix) = print_array(io, A.ψ)
function print_array(io::IO, A::ImputingMatrix)
    println(io, "W:")
    print_array(io, A.W)
    println(io, "\n\nψ:")
    _print_params(io, A)
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

_mul(A::RowImputingMatrix, ::AbstractVector{Missing}) = A.W * A.ψ
_mul(A::RowImputingMatrix, B::AbstractMatrix{Missing}) = repeat(A.W * A.ψ, 1, size(B, 2))
_mul(A::RowImputingMatrix, B::AbstractVecOrMat{Maybe{T}}) where T <: Number = A.W * _fill_in(A.ψ, B)

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

_mul(A::ColImputingMatrix, B::MaybeHotVector{Missing}) = A.ψ
_mul(A::ColImputingMatrix, B::MaybeHotMatrix{Missing}) = repeat(A.ψ, 1, size(B, 2))
function _mul(A::ColImputingMatrix{T}, B::MaybeHotMatrix{Maybe{U}}) where {T, U <: Integer}
    m = .!ismissing.(B.I)
    C = similar(B, T, size(A, 1), size(B, 2))
    C .= A.ψ
    C[:, m] .= B[:, skipmissing(B.I)]
    C
end
function rrule(::typeof(_mul), A::ColImputingMatrix{T}, B::MaybeHotMatrix{Maybe{U}}) where {T, U <: Integer}
    @error "TODO"
end

RowImputingDense(d::Dense) = Dense(RowImputingMatrix(d.W), d.b, d.σ)
RowImputingDense(args...) = RowImputingDense(Dense(args...))
ColImputingDense(d::Dense) = Dense(ColImputingMatrix(d.W), d.b, d.σ)
ColImputingDense(args...) = ColImputingDense(Dense(args...))

_name(::RowImputingMatrix) = "RowImputing"
_name(::ColImputingMatrix) = "ColImputing"
function Base.show(io::IO, l::Dense{F, <:ImputingMatrix}) where F
  print(io, "$(_name(l.W))Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end
