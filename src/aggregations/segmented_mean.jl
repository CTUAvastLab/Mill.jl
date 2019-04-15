struct SegmentedMean{T} <: AggregationFunction
    C::T
end

SegmentedMean(d::Int) = SegmentedMean(param(zeros(Float32, d)))
Flux.@treelike SegmentedMean

Base.show(io::IO, sm::SegmentedMean) = print(io, "SegmentedMean($(length(sm.C)))\n")
modelprint(io::IO, sm::SegmentedMean; pad=[]) = paddedprint(io, "SegmentedMean($(length(sm.C)))")

(m::SegmentedMean)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)
(m::SegmentedMean)(x, args...) = _mean_grad(x, m.C, args...)
(m::SegmentedMean)(::Missing, args...) = _mean_grad(missing, m.C, args...)

_mean_grad(args...) = Flux.Tracker.track(_mean_grad, args...)
_mean_grad(x::Matrix, c::Vector, bags) = segmented_mean(x, c, bags)
_mean_grad(x::Matrix, c::Vector, bags, w::Vector) = segmented_mean(x, c, bags, w)
_mean_grad(x::Matrix, c::Vector, bags, w::Vector, mask::Vector) = segmented_mean(x, c, bags, w, mask)
Flux.Tracker.@grad function _mean_grad(args...)
    n = segmented_mean(Flux.data.(args)...)
    grad = Δ -> segmented_mean_back(Δ, n, args...)
    n, grad
end

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

@generated function segmented_mean_back(Δ, n::Matrix, x::MaybeMatrix, C::AbstractVector, bags::AbstractBags, w::MaybeVector=nothing, mask::MaybeMask=nothing) 
    init_rule = quote Δ = Flux.data(Δ) end
    empty_bag_update_rule = @do_nothing
    bag_update_rule = @do_nothing
    after_bag_rule = @do_nothing
    mask_rule = @mask_rule mask
    return_tuple = Expr(:tuple)
    return_tuple.args = fill(nothing, 5)

    if w <: Nothing
        init_bag_rule = @do_nothing
    else
        init_bag_rule = :(ws = sum(@view w[b]))
    end

    if x <: Tracked
        push!(init_rule.args, :(x = Flux.data(x)))
        push!(init_rule.args, :(dx = similar(x)))
        push!(bag_update_rule.args, w <: Nothing ?
              :(dx[i, bi] = Δ[i, j] / length(b))
              :
              :(dx[i, bi] = w[bi] * Δ[i, j] / ws)
             )
        return_tuple.args[1] = :dx
    end

    if C <: Tracked
        push!(init_rule.args, :(C = Flux.data(C)))
        push!(init_rule.args, :(dC = zero(C)))
        push!(empty_bag_update_rule.args, :(dC[i] += Δ[i, j]))
        return_tuple.args[2] = :dC
    end

    if w <: Tracked
        push!(init_rule.args, :(w = Flux.data(w)))
        push!(init_rule.args, :(dw = zero(w)))
        push!(bag_update_rule.args, :(dw[bi] += Δ[i, j] * (x[i, bi] - n[i, j]) / ws))
        return_tuple.args[4] = :dw
    end

    return_rule = Expr(:return, return_tuple)
    return complete_body(init_rule, empty_bag_update_rule, init_bag_rule, mask_rule,
              bag_update_rule, after_bag_rule, return_rule)
end
