struct SegmentedSum{T, V <: AbstractVector{T}} <: AggregationFunction
    ψ::V
end

Flux.@functor SegmentedSum

_SegmentedSum(d::Int) = SegmentedSum(zeros(Float32, d))

function (m::SegmentedSum{T})(x::Maybe{AbstractMatrix{T}}, bags::AbstractBags,
                              w::Optional{AbstractVecOrMat{T}}=nothing) where T
    segmented_sum_forw(x, m.ψ, bags, w)
end
function (m::SegmentedSum{T})(x::AbstractMatrix{T}, bags::AbstractBags,
                              w::Optional{AbstractVecOrMat{T}}, mask::AbstractVector) where T
    segmented_sum_forw(x .* mask', m.ψ, bags, w)
end

segmented_sum_forw(::Missing, ψ::AbstractVector, bags::AbstractBags, w) = repeat(ψ, 1, length(bags))
function segmented_sum_forw(x::AbstractMatrix, ψ::AbstractVector, bags::AbstractBags, w::Optional{AbstractVecOrMat}) 
    y = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                y[i, bi] = ψ[i]
            end
        else
            for j in b
                for i in 1:size(x, 1)
                    y[i, bi] += weight(w, i, j, eltype(x)) * x[i, j]
                end
            end
        end
    end
    y
end

function segmented_sum_back(Δ, y, x, ψ, bags, w) 
    dx = similar(x)
    dψ = zero(ψ)
    dw = isnothing(w) ? Zero() : zero(w)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                dψ[i] += Δ[i, bi]
            end
        else
            for j in b
                for i in 1:size(x, 1)
                    dx[i, j] = weight(w, i, j, eltype(x)) * Δ[i, bi]
                    ∇dw_segmented_sum!(dw, Δ, x, y, w, i, j, bi)
                end
            end
        end
    end
    dx, dψ, DoesNotExist(), dw
end

function segmented_sum_back(Δ, y, x::Missing, ψ, bags, w::Nothing) 
    dψ = zero(ψ)
    @inbounds for (bi, b) in enumerate(bags)
        for i in eachindex(ψ)
            dψ[i] += Δ[i, bi]
        end
    end
    Zero(), dψ, DoesNotExist(), Zero()
end

∇dw_segmented_sum!(dw::Zero, Δ, x, y, w::Nothing, i, j, bi) = nothing
function ∇dw_segmented_sum!(dw::AbstractVector, Δ, x, y, w::AbstractVector, i, j, bi) 
    dw[j] += Δ[i, bi] * (x[i, j])
end
function ∇dw_segmented_sum!(dw::AbstractMatrix, Δ, x, y, w::AbstractMatrix, i, j, bi)
    dw[i, j] += Δ[i, bi] * (x[i, j])
end

function rrule(::typeof(segmented_sum_forw), args...)
    y = segmented_sum_forw(args...)
    grad = Δ -> (NO_FIELDS, segmented_sum_back(Δ, y, args...)...)
    y, grad
end
