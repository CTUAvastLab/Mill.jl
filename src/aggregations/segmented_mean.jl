struct SegmentedMean{T} <: AggregationFunction
    ψ::T
end

Flux.@functor SegmentedMean

_SegmentedMean(d::Int) = SegmentedMean(zeros(Float32, d))

(m::SegmentedMean)(x::MaybeMatrix, bags::AbstractBags, w=nothing) = segmented_mean_forw(x, m.ψ, bags, w)
function (m::SegmentedMean)(x::AbstractMatrix, bags::AbstractBags, w::AggregationWeights, mask::AbstractVector)
    segmented_mean_forw(x .* mask', m.ψ, bags, w)
end

segmented_mean_forw(::Missing, ψ::AbstractVector, bags::AbstractBags, w) = repeat(ψ, 1, length(bags))
function segmented_mean_forw(x::AbstractMatrix, ψ::AbstractVector, bags::AbstractBags, w::AggregationWeights) 
    y = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                y[i, bi] = ψ[i]
            end
        else
            for j in b
                for i in 1:size(x, 1)
                    y[i, bi] += weight(w, i, j) * x[i, j]
                end
            end
            y[:, bi] ./= bagnorm(w, b)
        end
    end
    y
end

function segmented_mean_back(Δ, y, x, ψ, bags, w) 
    dx = zero(x)
    dψ = zero(ψ)
    dw = isnothing(w) ? nothing : zero(w)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                dψ[i] += Δ[i, bi]
            end
        else
            ws = bagnorm(w, b)
            for j in b
                for i in 1:size(x, 1)
                    dx[i, j] += weight(w, i, j) * Δ[i, bi] / weightsum(ws, i)
                    ∇dw_segmented_mean!(dw, Δ, x, y, w, ws, i, j, bi)
                end
            end
        end
    end
    dx, dψ, nothing, dw
end

function segmented_mean_back(Δ, y, x::Missing, ψ, bags, w) 
    dψ = zero(ψ)
    @inbounds for (bi, b) in enumerate(bags)
        for i in eachindex(ψ)
            dψ[i] += Δ[i, bi]
        end
    end
    nothing, dψ, nothing, nothing
end

∇dw_segmented_mean!(dw::Nothing, Δ, x, y, w::Nothing, ws, i, j, bi) = nothing
function ∇dw_segmented_mean!(dw::AbstractVector, Δ, x, y, w::AbstractVector, ws, i, j, bi) 
    dw[j] += Δ[i, bi] * (x[i, j] - y[i, bi]) / ws
end
function ∇dw_segmented_mean!(dw::AbstractMatrix, Δ, x, y, w::AbstractMatrix, ws, i, j, bi)
    dw[i, j] += Δ[i, bi] * (x[i, j] - y[i, bi]) / ws[i]
end

Zygote.@adjoint function segmented_mean_forw(args...)
    y = segmented_mean_forw(args...)
    grad = Δ -> segmented_mean_back(Δ, y, args...)
    y, grad
end
