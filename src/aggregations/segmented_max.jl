struct SegmentedMax{T} <: AggregationFunction
    C::T
end

SegmentedMax(d::Int) = SegmentedMax(zeros(Float32, d))
Flux.@treelike SegmentedMax

Base.show(io::IO, sm::SegmentedMax) = print(io, "SegmentedMax($(length(sm.C)))\n")
modelprint(io::IO, sm::SegmentedMax; pad=[]) = paddedprint(io, "SegmentedMax($(length(sm.C)))")

(m::SegmentedMax)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)
(m::SegmentedMax)(x, args...) = segmented_max(x, m.C, args...)


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


function segmented_max_back(Δ, n, x, C, bags, w = nothing) 
    dx = zero(x)
    dC = zero(C)
    dw = (w == nothing) ? nothing : zero(w)
    v = similar(x, size(x, 1))
    idxs = zeros(Int, size(x, 1))
    for (j, b) in enumerate(bags)
        fill!(v, typemin(eltype(x)))
        if isempty(b)
            dC .+= @view Δ[:, j]
        else
            @inbounds for bi in b
              for i in 1:size(x,1)
                if v[i] < x[i, bi]
                  idxs[i] = bi
                  v[i] = x[i, bi]
                end
              end
            end
          for i in 1:size(x, 1)
            dx[i, idxs[i]] = Δ[i, j]
          end
        end
    end
    (dx, dC, nothing, dw)
end

Zygote.@adjoint function segmented_max(args...)
    n = segmented_max(args...)
    grad = Δ -> segmented_max_back(Δ, n, args...)
    n, grad
end