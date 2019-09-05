# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4908336/
struct SegmentedLSE{T, U} <: AggregationFunction
    p::T
    C::U
end

SegmentedLSE(d::Int) = SegmentedLSE(param(randn(Float32, d)), param(zeros(Float32, d)))
Flux.@treelike SegmentedLSE

Base.show(io::IO, n::SegmentedLSE) = print(io, "SegmentedLSE($(length(n.p)))\n")
modelprint(io::IO, n::SegmentedLSE; pad=[]) = paddedprint(io, "SegmentedLSE($(length(n.p)))")

(m::SegmentedLSE)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)
(m::SegmentedLSE)(x, args...) = __lse_grad(x, m.C, m.p, args...)

__lse_grad(x::Missing, args...) = _lse_grad(x, args...)
__lse_grad(x, args...) = let m = maximum(x, dims=2)
    m .+ _lse_grad(x .- m, args...)
end
_lse_grad(x::Union{Matrix, Missing}, C::Vector, p::Vector, bags) = segmented_lse(x, C, p, bags)
_lse_grad(x::Union{Matrix, Missing}, C::Vector, p::Vector, bags, w::Union{Vector, Nothing}) = segmented_lse(x, C, p, bags, w)
_lse_grad(x::Union{Matrix, Missing}, C::Vector, p::Vector, bags, w::Union{Vector, Nothing}, mask::Union{Vector, Nothing}) = segmented_lse(x, C, p, bags, w, mask)

Zygote.@adjoint function _lse_grad(args...)
    n = segmented_lse(args...)
    grad = Δ -> segmented_lse_back(Δ, n, args...)
    n, grad
end

@generated function segmented_lse(x::MaybeMatrix, C::AbstractVector, p::AbstractVector, bags::AbstractBags, w::MaybeVector=nothing, mask::MaybeMask=nothing) 
    x <: Missing && return @fill_missing
    init_rule = quote
        o = zeros(eltype(x), size(x, 1), length(bags))
    end
    empty_bag_update_rule = :(o[i, j] = C[i])
    mask_rule = @mask_rule mask
    init_bag_rule = @do_nothing
    bag_update_rule = :(o[i, j] += exp(p[i] * x[i, bi]))
    after_bag_rule = :(o[:, j] .= (log.(o[:, j]) .- log(length(b))) ./ p)
    return_rule = :(return o)
    return complete_body(init_rule, empty_bag_update_rule, init_bag_rule, mask_rule,
                         bag_update_rule, after_bag_rule, return_rule)
end

function segmented_lse_back(Δ, n, x, C, p, bags, w::MaybeVector=nothing, mask::MaybeMask=nothing)
    dx = zero(x)
    dp = zero(p)
    ss1 = zero(p)
    ss2 = zero(p)
    dC = zero(C)
    @inbounds for (j, b) in enumerate(bags)
        if isempty(b)
            dC .+= @view Δ[:, j]
        else
            ss1 .= ss2 .= 0
            for bi in b
                for i in 1:size(x,1)
                    e = exp(p[i] * x[i, bi])
                    dx[i, bi] = Δ[i, j] * e
                    ss1[i] += e
                    ss2[i] += x[i, bi] * e
                end
            end
            dx[:, b] ./= ss1
            dp .+= Δ[:, j] .* (ss2 ./ ss1 - n[:, j])
        end
    end
    (dx, dC, dp ./ p, nothing, nothing)
end
# @generated function segmented_lse_back(Δ, n::Matrix, x::MaybeMatrix, C::AbstractVector, p::AbstractVector, bags::AbstractBags, w::MaybeVector=nothing, mask::MaybeMask=nothing)
#     init_rule = quote Δ = Flux.data(Δ) end
#     empty_bag_update_rule = @do_nothing
#     bag_update_rule = quote
#         e = exp(p[i] * x[i, bi])
#     end
#     after_bag_rule = @do_nothing
#     mask_rule = @mask_rule mask
#     return_tuple = Expr(:tuple)
#     return_tuple.args = fill(nothing, 6)
#     init_bag_rule = @do_nothing


#     if x <: Tracked
#         push!(init_rule.args, quote
#                   x = Flux.data(x)
#                   dx = similar(x)
#                   ss1 = zero(p)
#               end)
#         push!(init_bag_rule.args, :(ss1 .= 0))
#         push!(bag_update_rule.args, quote
#                   dx[i, bi] = Δ[i, j] * e
#                   ss1[i] += e
#               end)
#         push!(after_bag_rule.args, :(dx[:, b] ./= ss1))
#         return_tuple.args[1] = :dx
#     end

#     if C <: Tracked
#         push!(init_rule.args, :(C = Flux.data(C)))
#         push!(init_rule.args, :(dC = zero(C)))
#         push!(empty_bag_update_rule.args, :(dC[i] += Δ[i, j]))
#         return_tuple.args[2] = :dC
#     end

#     if p <: Tracked
#         push!(init_rule.args, quote
#                   p = Flux.data(p)
#                   dp = zero(p)
#                   ss2 = zero(p)
#               end)
#         if !(x <: Tracked)
#             push!(init_rule.args, :(ss1 = zero(p)))
#             push!(init_bag_rule.args, :(ss1 .= 0))
#             push!(bag_update_rule.args, :(ss1[i] += e))
#         end
#         push!(init_bag_rule.args, :(ss2 .= 0))
#         push!(bag_update_rule.args, :(ss2[i] += x[i, bi] * e))
#         push!(after_bag_rule.args, :(dp .+= Δ[:, j] .* (ss2 ./ ss1 - n[:, j])))
#         return_tuple.args[3] = :(dp ./ p)
#     end

#     if w <: Tracked
#         error("Gradient w.r.t. w not defined")
#         # push!(init_rule.args, :(w = Flux.data(w)))
#         # push!(init_rule.args, :(dw = zero(w)))
#         # return_tuple.args[4] = :dw
#     end

#     return_rule = Expr(:return, return_tuple)
#     return complete_body(init_rule, empty_bag_update_rule, init_bag_rule, mask_rule,
#                          bag_update_rule, after_bag_rule, return_rule)
# end
