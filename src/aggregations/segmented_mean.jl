struct SegmentedMean{T} <: AggregationFunction
    C::T
end

Flux.@treelike SegmentedMean

SegmentedMean(d::Int) = SegmentedMean(zeros(Float32, d))

Base.show(io::IO, sm::SegmentedMean) = print(io, "SegmentedMean($(length(sm.C)))\n")
modelprint(io::IO, sm::SegmentedMean; pad=[]) = paddedprint(io, "SegmentedMean($(length(sm.C)))")

(m::SegmentedMean)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)
(m::SegmentedMean)(x, args...) = segmented_mean(x, m.C, args...)

segmented_mean(x, C, bags) = segmented_mean(x, C, bags, nothing, nothing)
segmented_mean(x, C, bags, w) = segmented_mean(x, C, bags, w, nothing)

function segmented_mean(x::Missing, C::AbstractVector, bags::AbstractBags, w, mask::Nothing)
    repeat(C, 1, length(bags))
end

function segmented_mean(x::AbstractMatrix, C::AbstractVector, bags::AbstractBags, w::AggregationWeights, mask::AbstractVector)
    segmented_mean(x .* mask', C, bags, w, nothing)
end

function segmented_mean(x::AbstractMatrix, C::AbstractVector, bags::AbstractBags, w::AggregationWeights, mask::Nothing) 
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

function segmented_mean_back(Δ, y, x, C, bags, w=nothing) 
    dx = similar(x)
    dC = zero(C)
    dw = (w == nothing) ? nothing : zero(w)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                dC[i] += C[i]
            end
        else
            ws = bagnorm(w, b)
            for j in b
                for i in 1:size(x, 1)
                    dx[i, j] = ∇dx_segmented_mean(Δ, w, ws, i, j, bi)
                    ∇dw_segmented_mean!(dw, Δ, x, y, w, ws, i, j, bi)
                end
            end
        end
    end
    (dx, dC, nothing, dw)
end

∇dx_segmented_mean(Δ, w::Nothing, ws, i, j, bi) = Δ[i, bi] / ws
∇dx_segmented_mean(Δ, w::AbstractVector, ws, i, j, bi) = w[j] * Δ[i, bi] / ws
∇dx_segmented_mean(Δ, w::AbstractMatrix, ws, i, j, bi) = w[i, j] * Δ[i, bi] / ws[i]

∇dw_segmented_mean!(dw::Nothing, Δ, x, y, w::Nothing, ws, i, j, bi) = nothing
function ∇dw_segmented_mean!(dw::AbstractVector, Δ, x, y, w::AbstractVector, ws, i, j, bi) 
    dw[j] += Δ[i, bi] * (x[i, j] - y[i, bi]) / ws
end
function ∇dw_segmented_mean!(dw::AbstractMatrix, Δ, x, y, w::AbstractMatrix, ws, i, j, bi)
    dw[i, j] += Δ[i, bi] * (x[i, j] - y[i, bi]) / ws[i]
end

Zygote.@adjoint function segmented_mean(args...)
    y = segmented_mean(args...)
    grad = Δ -> segmented_mean_back(Δ, y, args...)
    y, grad
end
