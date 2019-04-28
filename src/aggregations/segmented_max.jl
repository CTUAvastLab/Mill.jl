struct SegmentedMax{T} <: AggregationFunction
    C::T
end

SegmentedMax(d::Int) = SegmentedMax(param(zeros(Float32, d)))
Flux.@treelike SegmentedMax

Base.show(io::IO, sm::SegmentedMax) = print(io, "SegmentedMax($(length(sm.C)))\n")
modelprint(io::IO, sm::SegmentedMax; pad=[]) = paddedprint(io, "SegmentedMax($(length(sm.C)))")

(m::SegmentedMax)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)
(m::SegmentedMax)(x, args...) = _max_grad(x, m.C, args...)

_max_grad(args...) = Flux.Tracker.track(_max_grad, args...)
_max_grad(x::Union{Matrix, Missing}, C::Vector, bags) = segmented_max(x, C, bags)
_max_grad(x::Union{Matrix, Missing}, C::Vector, bags, w::Union{Vector, Nothing}) = segmented_max(x, C, bags, w)
_max_grad(x::Union{Matrix, Missing}, C::Vector, bags, w::Union{Vector, Nothing}, mask::Union{Vector, Nothing}) = segmented_max(x, C, bags, w, mask)

Flux.Tracker.@grad function _max_grad(args...)
    n = segmented_max(Flux.data.(args)...)
    grad = Δ -> segmented_max_back(Δ, n, args...)
    n, grad
end

@generated function segmented_max(x::MaybeMatrix, C::AbstractVector, bags::AbstractBags, w::MaybeVector=nothing, mask::MaybeMask=nothing) 
    x <: Missing && return @fill_missing
    init_rule = quote
        o = fill(typemin(eltype(x)), size(x, 1), length(bags))
    end
    empty_bag_update_rule = :(o[i, j] = C[i])
    init_bag_rule = @do_nothing
    mask_rule = @mask_rule mask
    if x <: Nothing
        bag_update_rule = @do_nothing
    else
        bag_update_rule = :(o[i, j] = max(o[i, j], x[i, bi]))
    end
    after_bag_rule = @do_nothing
    return_rule = :(return o)
    return complete_body(init_rule, empty_bag_update_rule, init_bag_rule, mask_rule,
                         bag_update_rule, after_bag_rule, return_rule)
end

@generated function segmented_max_back(Δ, n::Matrix, x::MaybeMatrix, C::AbstractVector, bags::AbstractBags, w::MaybeVector=nothing, mask::MaybeMask=nothing) 
    init_rule = quote Δ = Flux.data(Δ) end
    empty_bag_update_rule = @do_nothing
    init_bag_rule = @do_nothing
    bag_update_rule = @do_nothing
    after_bag_rule = @do_nothing
    mask_rule = @mask_rule mask
    return_tuple = Expr(:tuple)
    return_tuple.args = fill(nothing, 5)

    if x <: Tracked
        push!(init_rule.args, :(x = Flux.data(x)))
        push!(init_rule.args, :(dx = zero(x)))
        push!(init_rule.args, quote
                  v = similar(x, size(x, 1))
                  idxs = zeros(Int, size(x, 1))
              end)
        push!(init_bag_rule.args, :(fill!(v, typemin(eltype(x)))))
        push!(bag_update_rule.args, quote
                  if v[i] < x[i, bi]
                      idxs[i] = bi
                      v[i] = x[i, bi]
                  end
              end)
        push!(after_bag_rule.args, quote
                  for i in 1:size(x, 1)
                      dx[i, idxs[i]] = Δ[i, j]
                  end
              end)
        return_tuple.args[1] = :dx
    end

    if C <: Tracked
        push!(init_rule.args, :(C = Flux.data(C)))
        push!(init_rule.args, :(dC = zero(C)))
        push!(empty_bag_update_rule.args, :(dC[i] += Δ[i, j]))
        return_tuple.args[2] = :dC
    end

    if w <: Tracked
        push!(init_rule.args, :(dw = zero(w)))
        return_tuple.args[4] = :dw
    end

    return_rule = Expr(:return, return_tuple)
    return complete_body(init_rule, empty_bag_update_rule, init_bag_rule, mask_rule,
                         bag_update_rule, after_bag_rule, return_rule)
end
