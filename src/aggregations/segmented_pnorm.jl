# https://arxiv.org/pdf/1311.1780.pdf
struct SegmentedPNorm{T, U} <: AggregationFunction
    ρ::T
    c::T
    C::U
end

SegmentedPNorm(d::Int) = SegmentedPNorm(param(randn(Float32, d)), param(randn(Float32, d)), param(randn(Float32, d)))
Flux.@treelike SegmentedPNorm

p_map(ρ) = 1 .+ log.(1 .+ exp.(ρ))
inv_p_map(p) = log.(exp.(p-1) .- 1)

Base.show(io::IO, n::SegmentedPNorm) = print(io, "SegmentedPNorm($(length(n.ρ)))")
modelprint(io::IO, n::SegmentedPNorm; pad=[]) = paddedprint(io, "SegmentedPNorm($(length(n.ρ)))")

(m::SegmentedPNorm)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)
(m::SegmentedPNorm)(x, args...) = _pnorm_grad(x, m.C, m.ρ, m.c, args...)

_pnorm_grad(args...) = Flux.Tracker.track(_pnorm_grad, args...)
Flux.Tracker.@grad function _pnorm_grad(x, C, ρ, c, args...)
    n = segmented_pnorm(Flux.data(x), Flux.data(C), p_map(Flux.data(ρ)), Flux.data(c), Flux.data.(args)...)
    grad = Δ -> segmented_pnorm_back(Δ, n, x, C, p_map(ρ), ρ, c, args...)
    n, grad
end

@generated function segmented_pnorm(x::MaybeMatrix, C::AbstractVector, p::AbstractVector,
                                    c::AbstractVector, bags::AbstractBags, w::MaybeVector=nothing, mask::MaybeMask=nothing) 
    x <: Missing && return @fill_missing
    init_bag_rule = begin
        o = zeros(eltype(x), size(x, 1), length(bags))
    end
    empty_bag_update_rule = quote o[i, j] = C[i] end
    mask_rule = @mask_rule mask
    if w <: Nothing
        init_bag_rule = @do_nothing
        bag_update_rule = :(o[i, j] += abs(x[i, bi] - c[i]) ^ p[i] / length(b))
    else
        push!(init_rule.args, :(@assert all(w .> 0)))
        init_bag_rule = :(ws = sum(@view w[b]))
        bag_update_rule = :(o[i, j] += w[bi] * abs(x[i, bi] - c[i]) ^ p[i] / ws)
    end
    after_bag_rule = quote o[:, j] .^= 1 ./ p end
    return_rule = quote return o end
    return complete_body(init_rule, empty_bag_update_rule, init_bag_rule, mask_rule,
                         bag_update_rule, after_bag_rule, return_rule)
end

@generated function segmented_pnorm_back(Δ, n::Matrix, x::MaybeMatrix, C::AbstractVector, p::P, ρ::P, c::P, bags::AbstractBags, w::MaybeVector=nothing, mask::MaybeMask=nothing) where P <: AbstractVector
    init_rule = quote Δ = Flux.data(Δ) end
    empty_bag_update_rule = @do_nothing
    bag_update_rule = quote
        ab = abs(x[i, bi] - c[i])
        sig = sign(x[i, bi] - c[i])
    end
    after_bag_rule = @do_nothing
    mask_rule = @mask_rule mask
    return_tuple = Expr(:tuple)
    return_tuple.args = fill(nothing, 7)

    if w <: Nothing
        init_bag_rule = @do_nothing
    else
        init_bag_rule = quote ws = sum(@view w[b]) end
    end

    if x <: Tracked
        push!(init_rule.args, :(x = Flux.data(x)))
        push!(init_rule.args, :(dx = similar(x)))
        push!(bag_update_rule.args, w <: Nothing ? quote
                  dx[i, bi] = Δ[i, j] * sig
                  dx[i, bi] /= length(b)
                  dx[i, bi] *= (ab / n[i, j]) ^ (p[i] - 1)
              end : quote
                  dx[i, bi] = Δ[i, j] * w[bi] * sig
                  dx[i, bi] /= ws
                  dx[i, bi] *= (ab / n[i, j]) ^ (p[i] - 1)
              end
             )
        return_tuple.args[1] = :dx
    end

    if C <: Tracked
        push!(init_rule.args, quote 
                  C = Flux.data(C)
                  dC = zero(C)
              end)
        push!(empty_bag_update_rule.args, :(dC[i] += Δ[i, j]))
        return_tuple.args[2] = :dC
    end

    if p <: Tracked && ρ <: Tracked && c <: Tracked
        push!(init_rule.args, quote
                  p = Flux.data(p); ρ = Flux.data(ρ); c = Flux.data(c)
                  dp, dps1, dps2 = zero(p), zero(p), zero(p)
                  dc, dcs = zero(c), zero(c)
              end)
        push!(init_bag_rule.args, quote
                  dcs .= 0; dps1 .= 0; dps2 .= 0
              end)
        push!(bag_update_rule.args, w <: Nothing ? quote
                  dps1[i] += ab ^ p[i] * log(ab)
                  dps2[i] += ab ^ p[i]
                  dcs[i] -= sig * (ab ^ (p[i] - 1))
              end : quote
                  dps1[i] +=  w[bi] * ab ^ p[i] * log(ab)
                  dps2[i] +=  w[bi] * ab ^ p[i]
                  dcs[i] -= w[bi] * sig * (ab ^ (p[i] - 1))
              end)
        push!(after_bag_rule.args, w <: Nothing ? quote
                  t = n[:, j] ./ p .* (dps1 ./ dps2 .- (log.(dps2) .- log(max(1, length(b)))) ./ p)
                  dp .+= Δ[:, j] .* t
                  dcs ./= max(1, length(b))
                  dcs .*= n[:, j] .^ (1 .- p)
                  dc .+= Δ[:, j] .* dcs
              end : quote
                  t = n[:, j] ./ p .* (dps1 ./ dps2 .- (log.(dps2) .- log(ws)) ./ p)
                  dp .+= Δ[:, j] .* t
                  dcs ./= ws
                  dcs .*= n[:, j] .^ (1 .- p)
                  dc .+= Δ[:, j] .* dcs
              end)
        return_tuple.args[3] = :(dp .* σ.(ρ))
        return_tuple.args[4] = :dc
    end

    if w <: Tracked
        error("Gradient w.r.t. w not defined yet")
        # push!(init_rule.args, :(w = Flux.data(w)))
        # push!(init_rule.args, :(dw = zero(w)))
        # push!(bag_update_rule.args, :(dw[bi] += Δ[i, j] * (x[i, bi] - n[i, j]) / ws))
        # return_tuple.args[5] = :dw
    end

    return_rule = Expr(:return, return_tuple)
    return complete_body(init_rule, empty_bag_update_rule, init_bag_rule, mask_rule,
                         bag_update_rule, after_bag_rule, return_rule)
end
