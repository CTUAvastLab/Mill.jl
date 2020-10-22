# struct ImputingMatrix{T, U <: AbstractMatrix{T}, V <: AbstractVector{T}} <: AbstractMatrix{T}
struct ImputingMatrix{T, U <: AbstractMatrix{T}, V <: AbstractVector{T}}
    W::U
    ψ::V
end

Flux.@functor ImputingMatrix

ImputingMatrix(W::AbstractMatrix{T}) where T = ImputingMatrix(W, zeros(T, size(W, 1)))

Flux.@forward ImputingMatrix.W Base.size, Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex

function Base.print_matrix(io::IO, A::ImputingMatrix)
    println(io, "W:")
    print_matrix(io, A.W)
    println(io, "\nψ:")
    print_matrix(io, A.ψ)
end

function Flux.params!(p::Params, A::ImputingMatrix, seen=IdSet())
  A in seen && return
  push!(seen, A)
  push!(p, A.W, A.ψ)
end

Base.vcat(As::ImputingMatrix...) = ImputingMatrix(vcat((A.W for A in As)...), vcat((A.ψ for A in As)...))

_fill_in(::AbstractVector{T}, B::AbstractVecOrMat{T}) where T = B
function _fill_in(ψ::AbstractVector{T}, B::AbstractVecOrMat{<:MissingElement{T}}) where T
    X = similar(B, T)
    @inbounds for i in CartesianIndices(B)
        X[i] = ismissing(B[i]) ? ψ[i[1]] : B[i]
    end
    X
end

A::ImputingMatrix * B::AbstractVecOrMat{<:MissingElement{T}} where T = _imput_mult(A.W, A.ψ, B)
_imput_mult(W, ψ, B) = W * _fill_in(ψ, B)
