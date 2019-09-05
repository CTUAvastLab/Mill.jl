struct SegmentedMean{T} <: AggregationFunction
    C::T
end

SegmentedMean(d::Int) = SegmentedMean(zeros(Float32, d))
Flux.@treelike SegmentedMean

Base.show(io::IO, sm::SegmentedMean) = print(io, "SegmentedMean($(length(sm.C)))\n")
modelprint(io::IO, sm::SegmentedMean; pad=[]) = paddedprint(io, "SegmentedMean($(length(sm.C)))")

(m::SegmentedMean)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)
(m::SegmentedMean)(x, args...) = segmented_mean(x, m.C, args...)

@generated function segmented_mean(x::MaybeMatrix, C::AbstractVector, bags::AbstractBags, w::MaybeVector=nothing, mask::MaybeMask=nothing) 
    x <: Missing && return @fill_missing
    init_rule = quote
        o = zeros(eltype(x), size(x, 1), length(bags))
    end
    empty_bag_update_rule = :(o[i, j] = C[i])
    if w <: Nothing
        init_bag_rule = @do_nothing
        bag_update_rule = :(o[i, j] += x[i, bi] / length(b))
    else
        init_bag_rule = :(ws = sum(@view w[b]))
        bag_update_rule = :(o[i, j] += w[bi] * x[i, bi] / ws)
    end
    mask_rule = @mask_rule mask
    after_bag_rule = @do_nothing
    return_rule = :(return o)
    return complete_body(init_rule, empty_bag_update_rule, init_bag_rule, mask_rule,
                         bag_update_rule, after_bag_rule, return_rule)
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

bagnormalization(w::Nothing, b) = length(b)
bagnormalization(w, b) = sum(w[i] for i in b)

∇dx_segmented_mean(Δ, bi, i, j, w::Nothing, ws) = Δ[i, j] / ws
∇dx_segmented_mean(Δ, bi, i, j, w::AbstractVector, ws) = w[bi] * Δ[i, j] / ws
∇dx_segmented_mean(Δ, bi, i, j, w::AbstractMatrix, ws) = w[i, bi] * Δ[i, j] / ws

∇dx_segmented_mean!(dw::Nothing, bi, x, Δ, n, i, j, w::Nothing, ws) = nothing
∇dx_segmented_mean!(dw::Vector, bi, x, Δ, n, i, j, w::Vector, ws) = dw[bi] += Δ[i, j] * (x[i, bi] - n[i, j]) / ws

