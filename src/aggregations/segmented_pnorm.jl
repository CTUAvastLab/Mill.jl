# https://arxiv.org/pdf/1311.1780.pdf
struct SegmentedPNorm{T, U, V} <: AggregationFunction
    ρ::T
    c::U
    C::V
end

SegmentedPNorm(d::Int) = SegmentedPNorm(randn(Float32, d), randn(Float32, d), zeros(Float32, d))
Flux.@treelike SegmentedPNorm

p_map(ρ) = 1 .+ log.(1 .+ exp.(ρ))
inv_p_map(p) = log.(exp.(p-1) .- 1)

Base.show(io::IO, n::SegmentedPNorm) = print(io, "SegmentedPNorm($(length(n.ρ)))\n")
modelprint(io::IO, n::SegmentedPNorm; pad=[]) = paddedprint(io, "SegmentedPNorm($(length(n.ρ)))")

(m::SegmentedPNorm)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)
(m::SegmentedPNorm)(x, args...) = _pnorm_grad(x, m.C, m.ρ, m.c, args...)

_pnorm_grad(x::Union{Matrix, Missing}, C::Vector, ρ::Vector, c::Vector, bags) = segmented_pnorm(x, C, p_map(ρ), c, bags)
_pnorm_grad(x::Union{Matrix, Missing}, C::Vector, ρ::Vector, c::Vector, bags, w::Union{Vector, Nothing}) = segmented_pnorm(x, C, p_map(ρ), c, bags, w)
_pnorm_grad(x::Union{Matrix, Missing}, C::Vector, ρ::Vector, c::Vector, bags, w::Union{Vector, Nothing}, mask::Union{Vector, Nothing}) = segmented_pnorm(x, C, p_map(ρ), c, bags, w, mask)

Zygote.@adjoint function _pnorm_grad(x, C, ρ, c, args...)
    n = segmented_pnorm(x, C, p_map(ρ), c, args...)
    grad = Δ -> segmented_pnorm_back(Δ, n, x, C, p_map(ρ), ρ, c, args...)
    n, grad
end

@generated function segmented_pnorm(x::MaybeMatrix, C::AbstractVector, p::AbstractVector,
                                    c::AbstractVector, bags::AbstractBags, w::MaybeVector=nothing, mask::MaybeMask=nothing) 
    x <: Missing && return @fill_missing
    init_rule = quote
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

segmented_pnorm_back(Δ, n::Matrix, x::MaybeMatrix, C::AbstractVector, p::AbstractVector, ρ::AbstractVector, c::AbstractVector, bags::AbstractBags, w::Nothing, mask::MaybeMask=nothing) = segmented_pnorm_back(Δ, n, x, C, p, ρ, c, bags, Ones(size(x,2)), mask)

function segmented_pnorm_back(Δ, n::Matrix, x::MaybeMatrix, C::AbstractVector, p::AbstractVector, ρ::AbstractVector, c::AbstractVector, bags::AbstractBags, w::AbstractVector = Ones(size(x,2)), mask::MaybeMask = nothing)
    dp = zero(p)
    dps1 = zero(p)
    dps2 = zero(p)
    dc = zero(c)
    dC = zero(C)
    dcs = zero(c)
    dx = similar(x)
    @inbounds for (j, b) in enumerate(bags)
        if isempty(b)
            dC .+= @view Δ[:, j]
        else
          ws = bagnormalization(w, b)
          dcs .= 0
          dps1 .= 0
          dps2 .= 0
          for bi in b
              for i in 1:size(x,1)
                  ab = abs(x[i, bi] - c[i])
                  sig = sign(x[i, bi] - c[i])
                  dps1[i] +=  w[bi] * ab ^ p[i] * log(ab)
                  dps2[i] +=  w[bi] * ab ^ p[i]
                  dcs[i] -= w[bi] * sig * (ab ^ (p[i] - 1))
                  dx[i, bi] = Δ[i, j] * w[bi] * sig
                  dx[i, bi] /= ws
                  dx[i, bi] *= (ab / n[i, j]) ^ (p[i] - 1)
              end
          end
          t = n[:, j] ./ p .* (dps1 ./ dps2 .- (log.(dps2) .- log(ws)) ./ p)
          dp .+= Δ[:, j] .* t
          dcs ./= ws
          dcs .*= n[:, j] .^ (1 .- p)
          dc .+= Δ[:, j] .* dcs
        end
    end
    dρ = dp .* σ.(ρ)
    (dx, dC, nothing, dρ, dc, nothing, nothing, nothing)
end