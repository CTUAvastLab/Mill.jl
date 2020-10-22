struct ImputingMatrix{T, U <: AbstractMatrix{T}, V <: AbstractVector{T}} <: AbstractMatrix{T}
    W::U
    ψ::V
end

Flux.@functor ImputingMatrix

ImputingMatrix(W::AbstractMatrix{T}) where T = ImputingMatrix(W, zeros(T, size(W, 1)))

Flux.@forward ImputingMatrix.W Base.size, Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex

Base.vcat(As::ImputingMatrix...) = ImputingMatrix(vcat((A.W for A in As)...), vcat((A.ψ for A in As)...))

_fill_in(::ImputingMatrix{T}, B::AbstractVecOrMat{T}) where T = B
function _fill_in(A::ImputingMatrix{T}, B::AbstractVecOrMat{<:MissingElement{T}}) where T
    X = similar(B, T)
    @inbounds for i in CartesianIndices(B)
        X[i] = ismissing(B[i]) ? A.ψ[i[1]] : B[i]
    end
    X
end

A::ImputingMatrix * B::AbstractVecOrMat{<:MissingElement{T}} where T = A.W * _fill_in(A, B)
