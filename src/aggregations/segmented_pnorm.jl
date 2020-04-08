# https://arxiv.org/pdf/1311.1780.pdf
struct SegmentedPNorm{T, U, V} <: AggregationFunction
    ρ::T
    c::U
    C::V
end

Flux.@functor SegmentedPNorm

SegmentedPNorm(d::Int) = SegmentedPNorm(randn(Float32, d), randn(Float32, d), zeros(Float32, d))

p_map(ρ) = 1 .+ softplus.(ρ)
inv_p_map = (p) -> max.(p .- 1, 0) .+ log1p.(-exp.(-abs.(p .- 1)))

(m::SegmentedPNorm)(x::Missing, bags::AbstractBags, w=nothing) = segmented_pnorm_forw(x, m.C, nothing, bags, w)
(m::SegmentedPNorm)(x::AbstractMatrix, bags::AbstractBags, w=nothing) = segmented_pnorm_forw(x .- m.c, m.C, p_map(m.ρ), bags, w)
function (m::SegmentedPNorm)(x::AbstractMatrix, bags::AbstractBags, w::AggregationWeights, mask::AbstractVector)
    segmented_pnorm_forw((x.-m.c) .* mask', m.C, p_map(m.ρ), bags, w)
end

function _pnorm_precomp(x::AbstractMatrix, bags)
    M = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if !isempty(b)
            for i in 1:size(x, 1)
                M[i, bi] = abs(x[i, first(b)])
            end
            for j in b[2:end]
                for i in 1:size(x, 1)
                    M[i, bi] = max(M[i, bi], abs(x[i, j]))
                end
            end
        end
    end
    M
end

function _segmented_pnorm_norm(a::AbstractMatrix, C::AbstractVector, p::AbstractVector, bags::AbstractBags, w, M) 
    isnothing(w) || @assert all(w .> 0)
    y = zeros(eltype(a), size(a, 1), length(bags))
    M = _pnorm_precomp(a, bags)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                y[i, bi] = C[i]
            end
        else
            ws = bagnorm(w, b)
            for j in b
                for i in 1:size(a, 1)
                    y[i, bi] += weight(w, i, j) * abs(a[i, j]/M[i, bi]) ^ p[i]
                end
            end
            for i in 1:size(a, 1)
                y[i, bi] = M[i, bi] * (y[i, bi] / weightsum(ws, i))^(1/p[i])
            end
        end
    end
    y
end

segmented_pnorm_forw(::Missing, C::AbstractVector, p, bags::AbstractBags, w) = repeat(C, 1, length(bags))
function segmented_pnorm_forw(a::MaybeMatrix, C::AbstractVector, p::AbstractVector, bags::AbstractBags, w) 
    M = _pnorm_precomp(a, bags)
    _segmented_pnorm_norm(a, C, p, bags, w, M)
end

function segmented_pnorm_back(Δ, y, a::AbstractMatrix, C, p, bags::AbstractBags, w::AggregationWeights, M)
    da = similar(a)
    dp = zero(p)
    dps1 = zero(p)
    dps2 = zero(p)
    dC = zero(C)
    dw = isnothing(w) ? nothing : zero(w)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                dC[i] += Δ[i, bi]
            end
        else
            ws = bagnorm(w, b)
            dps1 .= 0
            dps2 .= 0
            for j in b
                for i in 1:size(a, 1)
                    ab = abs(a[i, j])
                    da[i, j] = Δ[i, bi] * weight(w, i, j) * sign(a[i, j]) / weightsum(ws, i) 
                    da[i, j] *= (ab / y[i, bi]) ^ (p[i] - 1)
                    ww = weight(w, i, j) * (ab / M[i, bi]) ^ p[i]
                    dps1[i] +=  ww * log(ab)
                    dps2[i] +=  ww
                end
            end
            for i in 1:size(a, 1)
                t = y[i, bi] / p[i]
                t *= dps1[i] / dps2[i] - (p[i] * log(M[i, bi]) + log(dps2[i]) - log(weightsum(ws, i))) / p[i]
                dp[i] += Δ[i, bi] * t
            end
        end
    end
    da, dC, dp, nothing, dw
end

function segmented_pnorm_back(Δ, y, C, bags) 
    dC = zero(C)
    @inbounds for (bi, b) in enumerate(bags)
        for i in eachindex(C)
            dC[i] += Δ[i, bi]
        end
    end
    nothing, dC, nothing, nothing, nothing
end

∇dw_segmented_pnorm!(dw::Nothing, Δ, a, y, w::Nothing, ws, i, j, bi) = error("Not implemented yet!")
function ∇dw_segmented_pnorm!(dw::AbstractVector, Δ, a, y, w::AbstractVector, ws, i, j, bi) 
    error("Not implemented yet!")
end
function ∇dw_segmented_pnorm!(dw::AbstractMatrix, Δ, a, y, w::AbstractMatrix, ws, i, j, bi)
    error("Not implemented yet!")
end

Zygote.@adjoint function segmented_pnorm_forw(a::AbstractMatrix, C, p, bags, w)
    M = _pnorm_precomp(a, bags)
    y = _segmented_pnorm_norm(a, C, p, bags, w, M)
    grad = Δ -> segmented_pnorm_back(Δ, y, a, C, p, bags, w, M)
    y, grad
end

Zygote.@adjoint function segmented_pnorm_forw(a::Missing, C, p, bags, w)
    y = segmented_pnorm_forw(a, C, p, bags, w)
    grad = Δ -> segmented_pnorm_back(Δ, y, C, bags)
    y, grad
end
