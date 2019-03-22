struct SegmentedMean{T} <: AggregationFunction
    C::T
end

SegmentedMean(d::Int) = SegmentedMean(param(zeros(Float32, d)))
Flux.@treelike SegmentedMean

modelprint(io::IO, sm::SegmentedMean; pad=[]) = paddedprint(io, "SegmentedMean($(length(sm.C)))")

# FORWARD PASS
(m::SegmentedMean)(x, args...) = segmented_mean(x, m.C, args...)
(m::SegmentedMean)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)
(m::SegmentedMean{<:TrackedVector})(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)

@generated function segmented_mean(x::MaybeMatrix, C::Vector, bags::AbstractBags, w::MaybeVector=nothing, mask::MaybeMask=nothing) 
    init_rule = Expr(:block, quote 
                         o = zeros(eltype(x), size(x, 1), length(bags))
                     end)
    empty_bag_update_rule = :(o[i, j] = C[i])
    if !(w <: Nothing)
        init_bag_rule = :(ws = sum(@view w[b]))
    else
        init_bag_rule = @do_nothing
    end
    mask_rule = @mask_rule mask
    if (w <: Nothing)
        bag_update_rule = :(o[i, j] += x[i, bi] / length(b))
    else
        bag_update_rule = :(o[i, j] += w[bi] * x[i, bi] / ws)
    end
    after_bag_rule = @do_nothing
    return_rule = :(return o)
    return complete_body(init_rule, empty_bag_update_rule, init_bag_rule, mask_rule,
              bag_update_rule, after_bag_rule, return_rule)
end

# BACKWARD
(m::SegmentedMean{<:AbstractVector})(x::TrackedMatrix, args...) = _mean_grad(x, m.C, args...)
(m::SegmentedMean{<:TrackedVector})(x, args...) = _mean_grad(x, m.C, args...)
(m::SegmentedMean{<:TrackedVector})(x::TrackedMatrix, args...) = _mean_grad(x, m.C, args...)

_mean_grad(x, C, args...) = Flux.Tracker.track(_mean_grad, x, C, args...)
Flux.Tracker.@grad function _mean_grad(x, C, args...)
    n = segmented_mean(Flux.data(x), Flux.data(C), Flux.data.(args)...)
    grad = Δ -> segmented_mean_back(Δ, n, x, C, args...)
    n, grad
end

@generated function segmented_mean_back(Δ, n::Matrix, x::MaybeInputMatrix, C::InputVector, bags::AbstractBags, w::MaybeInputVector=nothing, mask::MaybeMask=nothing) 
    init_rule = quote Δ = Flux.data(Δ) end
    empty_bag_update_rule = Expr(:block)
    bag_update_rule = Expr(:block)
    after_bag_rule = @do_nothing
    mask_rule = @mask_rule mask
    return_tuple = Expr(:tuple)
    return_tuple.args = fill(nothing, 5)

    if !(w <: Nothing)
        init_bag_rule = :(ws = sum(@view w[b]))
    else
        init_bag_rule = @do_nothing
    end

    if (x <: Tracked)
        push!(init_rule.args, :(x = Flux.data(x)))
        push!(init_rule.args, :(dx = similar(x)))
        push!(bag_update_rule.args, w <: Nothing ?
              :(dx[i, bi] = Δ[i, j] / length(b))
              :
              :(dx[i, bi] = w[bi] * Δ[i, j] / ws)
             )
        return_tuple.args[1] = :dx
    end

    if (C <: Tracked)
        push!(init_rule.args, :(C = Flux.data(C)))
        push!(init_rule.args, :(dC = zero(C)))
        push!(empty_bag_update_rule.args, :(dC[i] += Δ[i, j]))
        return_tuple.args[2] = :dC
    end

    if (w <: Tracked)
        push!(init_rule.args, :(w = Flux.data(w)))
        push!(init_rule.args, :(dw = zero(w)))
        push!(bag_update_rule.args, :(dw[bi] += Δ[i, j] * (x[i, bi] - n[i, j]) / ws))
        return_tuple.args[4] = :dw
    end

    return_rule = Expr(:return, return_tuple)
    return complete_body(init_rule, empty_bag_update_rule, init_bag_rule, mask_rule,
              bag_update_rule, after_bag_rule, return_rule)
end
