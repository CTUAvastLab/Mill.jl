struct SegmentedMean{T} <: AggregationFunction
    C::T
end

Flux.@treelike SegmentedMean

SegmentedMean(d::Int) = SegmentedMean(zeros(Float32, d))

Base.show(io::IO, sm::SegmentedMean) = print(io, "SegmentedMean($(length(sm.C)))\n")
modelprint(io::IO, sm::SegmentedMean; pad=[]) = paddedprint(io, "SegmentedMean($(length(sm.C)))")

(m::SegmentedMean)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)
(m::SegmentedMean)(x, args...) = segmented_mean(x, m.C, args...)

function segmented_mean(x::Missing, C::AbstractVector, bags::AbstractBags, w = nothing, mask = nothing)
    repeat(C, 1, length(bags))
end

function segmented_mean(x::AbstractMatrix, C::AbstractVector, bags::AbstractBags, w = Fill(true, size(x,2)), mask = Fill(true, size(x,2))) 
    o = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (j, b) in enumerate(bags)
        if isempty(b)
            for i in 1:size(x, 1)
                o[i, j] = C[i]
            end
        else
            for bi in b
               for i in 1:size(x, 1)
                    o[i, j] += w[bi] * x[i, bi]
                end
            end
            o[:, j] ./= bagnormalization(w, b)
        end
    end
    o
end

function segmented_mean_back(Δ, n, x, C, bags, w = nothing) 
    dx = similar(x)
    dC = zero(C)
    dw = (w == nothing) ? nothing : zero(w)
    for (j, b) in enumerate(bags)
        if isempty(b)
            dC .+= @view Δ[:, j]
        else
            ws = bagnormalization(w, b)
            @inbounds for bi in b
                for i in 1:size(x, 1)
                    dx[i, bi] = ∇dx_segmented_mean(Δ, bi, i, j, w, ws)
                    ∇dx_segmented_mean!(dw, bi, x, Δ, n, i, j, w, ws)
                end
            end
        end
    end
    (dx, dC, nothing, dw)
end

Zygote.@adjoint function segmented_mean(args...)
    n = segmented_mean(args...)
    grad = Δ -> segmented_mean_back(Δ, n, args...)
    n, grad
end

∇dx_segmented_mean(Δ, bi, i, j, w::Nothing, ws) = Δ[i, j] / ws
∇dx_segmented_mean(Δ, bi, i, j, w::AbstractVector, ws) = w[bi] * Δ[i, j] / ws
∇dx_segmented_mean(Δ, bi, i, j, w::AbstractMatrix, ws) = w[i, bi] * Δ[i, j] / ws[i]

∇dx_segmented_mean!(dw::Nothing, bi, x, Δ, n, i, j, w::Nothing, ws) = nothing
∇dx_segmented_mean!(dw::AbstractVector, bi, x, Δ, n, i, j, w::AbstractVector, ws) = dw[bi] += Δ[i, j] * (x[i, bi] - n[i, j]) / ws
∇dx_segmented_mean!(dw::AbstractMatrix, bi, x, Δ, n, i, j, w::AbstractMatrix, ws) = dw[i, bi] += Δ[i, j] * (x[i, bi] - n[i, j]) / ws[i]

