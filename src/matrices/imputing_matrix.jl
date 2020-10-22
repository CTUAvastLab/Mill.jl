struct ImputingMatrix{T, U <: AbstractMatrix{T}, V <: AbstractVector{T}} <: AbstractMatrix{T}
    W::U
    ψ::V
end

Flux.@functor ImputingMatrix

ImputingMatrix(W::AbstractMatrix{T}) where T <: Real = ImputingMatrix(W, zeros(T, size(W, 1)))

Flux.@forward ImputingMatrix.W Base.size, Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex

Base.vcat(As::ImputingMatrix...) = ImputingMatrix(vcat((A.W for A in As)...), vcat((A.ψ for A in As)...))

A::ImputingMatrix * B::AbstractMatOrVec{<:Real} = A.W * B
A::ImputingMatrix * B::AbstractMatOrVec{<:MissingElement{T}} where T <: Real = _imput_mult(A, B)

function _imput_mult(A::ImputingMatrix{T}, B::AbstractMatOrVec{<:MissingElement{T}}) where T <: Real
    X = similar(B, T)
    @inbounds for i in CartesianIndices(B)
        X[i] = ismissing(B[i]) ? A.ψ[i[1]] : B[i]
    end
    A * X
end
