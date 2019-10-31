struct SegmentedMean{T} <: AggregationFunction
    C::T
end

Flux.@treelike SegmentedMean
# Flux.@functor SegmentedMean

SegmentedMean(d::Int) = SegmentedMean(zeros(Float32, d))

Base.show(io::IO, sm::SegmentedMean) = print(io, "SegmentedMean($(length(sm.C)))\n")
modelprint(io::IO, sm::SegmentedMean; pad=[]) = paddedprint(io, "SegmentedMean($(length(sm.C)))")

(m::SegmentedMean)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)
(m::SegmentedMean)(x::MaybeMatrix, bags::AbstractBags, w=nothing) = segmented_mean_forw(x, m.C, bags, w)
function (m::SegmentedMean)(x::AbstractMatrix, bags::AbstractBags, w::AggregationWeights, mask::AbstractVector)
    segmented_mean_forw(x .* mask', m.C, bags, w)
end

segmented_mean_forw(::Missing, C::AbstractVector, bags::AbstractBags, w) = repeat(C, 1, length(bags))
function segmented_mean_forw(x::AbstractMatrix, C::AbstractVector, bags::AbstractBags, w::AggregationWeights) 
    y = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                y[i, bi] = C[i]
            end
        else
            for j in b
                for i in 1:size(x, 1)
                    y[i, bi] += weight(w, i, j)  * x[i, j]
                end
            end
            y[:, bi] ./= bagnorm(w, b)
        end
    end
    y
end

function segmented_mean_back(Δ, y, x, C, bags, w) 
    dx = similar(x)
    dC = zero(C)
    dw = isnothing(w) ? nothing : zero(w)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                dC[i] += Δ[i, bi]
            end
        else
            ws = bagnorm(w, b)
            for j in b
                for i in 1:size(x, 1)
                    dx[i, j] = weight(w, i, j) * Δ[i, bi] / weightsum(ws, i)
                    ∇dw_segmented_mean!(dw, Δ, x, y, w, ws, i, j, bi)
                end
            end
        end
    end
    dx, dC, nothing, dw
end

function segmented_mean_back(Δ, y, x::Missing, C, bags, w::Nothing) 
    dC = zero(C)
    @inbounds for (bi, b) in enumerate(bags)
        for i in eachindex(C)
            dC[i] += Δ[i, bi]
        end
    end
    nothing, dC, nothing, nothing
end

∇dw_segmented_mean!(dw::Nothing, Δ, x, y, w::Nothing, ws, i, j, bi) = nothing
function ∇dw_segmented_mean!(dw::AbstractVector, Δ, x, y, w::AbstractVector, ws, i, j, bi) 
    dw[j] += Δ[i, bi] * (x[i, j] - y[i, bi]) / ws
end
function ∇dw_segmented_mean!(dw::AbstractMatrix, Δ, x, y, w::AbstractMatrix, ws, i, j, bi)
    dw[i, j] += Δ[i, bi] * (x[i, j] - y[i, bi]) / ws[i]
end

@adjoint function segmented_mean_forw(args...)
    y = segmented_mean_forw(args...)
    grad = Δ -> segmented_mean_back(Δ, y, args...)
    y, grad
end
