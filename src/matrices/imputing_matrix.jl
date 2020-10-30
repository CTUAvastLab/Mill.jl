struct ImputingMatrix{T <: Number, R <: Optional{AbstractVector{T}},
                      C <: Optional{AbstractVector{T}}, U <: AbstractMatrix{T}} <: AbstractMatrix{T}
    W::U
    ψr::R
    ψc::C
end

const RowImputingMatrix{T <: Number, R <: AbstractVector{T}, U <: AbstractMatrix{T}} =
    ImputingMatrix{T, R, Nothing, U}
const ColImputingMatrix{T <: Number, C <: AbstractVector{T}, U <: AbstractMatrix{T}} =
    ImputingMatrix{T, Nothing, C, U}

Flux.@functor ImputingMatrix

ImputingMatrix(W::AbstractMatrix{T}, ψr=zeros(T, size(W, 2)), ψc=zeros(T, size(W, 1))) where T = ImputingMatrix(W, ψc, ψc)
RowImputingMatrix(W::AbstractMatrix{T}, ψr=zeros(T, size(W, 2))) where T = ImputingMatrix(W, ψr, nothing)
ColImputingMatrix(W::AbstractMatrix{T}, ψc=zeros(T, size(W, 1))) where T = ImputingMatrix(W, nothing, ψc)

Flux.@forward ImputingMatrix.W Base.size, Base.length, Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex

Base.hcat(As::Vararg{RowImputingMatrix}) = RowImputingMatrix(hcat((A.W for A in As)...), hcat((A.ψr for A in As)...))
Base.vcat(As::Vararg{ColImputingMatrix}) = ColImputingMatrix(vcat((A.W for A in As)...), vcat((A.ψc for A in As)...))

function print_array(io::IO, A::ImputingMatrix)
    println(io, "W:")
    print_array(io, A.W)
    if !isnothing(A.ψr)
        println(io, "\n\nRow ψ:")
        print_array(io, A.ψr')
    end
    if !isnothing(A.ψc)
        println(io, "\n\nCol ψ:")
        print_array(io, A.ψc)
    end
end

Base.push!(p::Params, A::ImputingMatrix) = push!(p, A.W, A.ψr, A.ψc)
Base.push!(p::Params, A::RowImputingMatrix) = push!(p, A.W, A.ψr)
Base.push!(p::Params, A::ColImputingMatrix) = push!(p, A.W, A.ψc)
function Flux.params!(p::Params, A::ImputingMatrix, seen=IdSet())
    A in seen && return
    push!(seen, A)
    push!(p, A)
end

A::ImputingMatrix * B::AbstractVector = _mul(A, B)
Zygote.@adjoint A::ImputingMatrix * B::AbstractVector = Zygote.pullback(_mul, A, B)
A::ImputingMatrix * B::AbstractMatrix = _mul(A, B)
Zygote.@adjoint A::ImputingMatrix * B::AbstractMatrix = Zygote.pullback(_mul, A, B)

_mul(A::ImputingMatrix, B::AbstractVecOrMat) = A.W * B
_mul(A::RowImputingMatrix, ::AbstractVector{Missing}) = A.W * A.ψr
_mul(A::RowImputingMatrix, B::AbstractMatrix{Missing}) = repeat(A.W * A.ψr, 1, size(B, 2))
_mul(A::RowImputingMatrix, B::AbstractVecOrMat{Maybe{T}}) where T <: Number = A.W * _fill_in(A.ψr, B)

_fill_in(ψr, B) = _fill_mask(ψr, B)[1]
function rrule(::typeof(_fill_in), ψr, B)
    X, m = _fill_mask(ψr, B)
    _fill_in_dψr(Δ) = (dψr = deepcopy(Δ); dψr[m] .= 0; sum(dψr, dims=2))
    _fill_in_dB(Δ) = (dB = deepcopy(Δ); dB[.!m] .= 0; dB)
    X, Δ -> (NO_FIELDS, @thunk(_fill_in_dψr(Δ)), @thunk(_fill_in_dB(Δ)))
end

function _fill_mask(ψr::AbstractVector{T}, B) where T
    m = .!ismissing.(B)
    X = similar(B, T)
    X .= ψr
    X[m] = B[m]
    X, m
end

ImputingDense(d::Dense) = Dense(ImputingMatrix(d.W), d.b, d.σ)
ImputingDense(args...) = ImputingDense(Dense(args...))
RowImputingDense(d::Dense) = Dense(RowImputingMatrix(d.W), d.b, d.σ)
RowImputingDense(args...) = RowImputingDense(Dense(args...))
ColImputingDense(d::Dense) = Dense(ColImputingMatrix(d.W), d.b, d.σ)
ColImputingDense(args...) = ColImputingDense(Dense(args...))

_name(::ImputingMatrix) = "Imputing"
_name(::RowImputingMatrix) = "RowImputing"
_name(::ColImputingMatrix) = "ColImputing"
function Base.show(io::IO, l::Dense{F, <:ImputingMatrix}) where F
  print(io, "$(_name(l.W))Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end
