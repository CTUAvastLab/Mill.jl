# https://arxiv.org/pdf/1311.1780.pdf
struct SegmentedPNorm{T, U, V} <: AggregationFunction
    ρ::T
    c::U
    C::V
end

Flux.@treelike SegmentedPNorm

SegmentedPNorm(d::Int) = SegmentedPNorm(randn(Float32, d), randn(Float32, d), zeros(Float32, d))

Base.show(io::IO, n::SegmentedPNorm) = print(io, "SegmentedPNorm($(length(n.ρ)))\n")
modelprint(io::IO, n::SegmentedPNorm; pad=[]) = paddedprint(io, "SegmentedPNorm($(length(n.ρ)))")

p_map(ρ) = 1 .+ softplus.(ρ)

(m::SegmentedPNorm)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)
(m::SegmentedPNorm)(x::Missing, bags::AbstractBags, w::Nothing=nothing) = segmented_pnorm_forw(x, m.C, nothing, bags, nothing)
(m::SegmentedPNorm)(x::AbstractMatrix, bags::AbstractBags, w=nothing) = segmented_pnorm_forw(x .- m.c, m.C, p_map(m.ρ), bags, w)
function (m::SegmentedPNorm)(x::AbstractMatrix, bags::AbstractBags, w::AggregationWeights, mask::AbstractVector)
    segmented_pnorm_forw((x.-m.c) .* mask', m.C, p_map(m.ρ), bags, w)
end

segmented_pnorm_forw(::Missing, C::AbstractVector, p, bags::AbstractBags, w) = repeat(C, 1, length(bags))
function segmented_pnorm_forw(a::AbstractMatrix, C::AbstractVector, p::AbstractVector, bags::AbstractBags, w) 
    isnothing(w) || @assert all(w .> 0)
    y = zeros(eltype(a), size(a, 1), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                y[i, bi] = C[i]
            end
        else
            ws = bagnorm(w, b)
            for j in b
                for i in 1:size(a, 1)
                    y[i, bi] += weight(w, i, j) * abs(a[i, j]) ^ p[i]
                end
            end
            for i in 1:size(a, 1)
                y[i, bi] /= weightsum(ws, i)
                y[i, bi] ^= 1 / p[i]
            end
        end
    end
    y
end

function segmented_pnorm_back(Δ, y, a::AbstractMatrix, C, p, bags::AbstractBags, w::AggregationWeights)
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
                    ww = weight(w, i, j) * ab ^ p[i]
                    dps1[i] +=  ww * log(ab)
                    dps2[i] +=  ww
                end
            end
            for i in 1:size(a, 1)
                t = y[i, bi] / p[i]
                t *= dps1[i] / dps2[i] - (log(dps2[i]) - log(weightsum(ws, i))) / p[i]
                dp[i] += Δ[i, bi] * t
            end
        end
    end
    da, dC, dp, nothing, dw
end

function segmented_pnorm_back(Δ, y, a::Missing, C, p, bags, w::Nothing) 
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

Zygote.@adjoint function segmented_pnorm_forw(a, C, p, args...)
    y = segmented_pnorm_forw(a, C, p, args...)
    grad = Δ -> segmented_pnorm_back(Δ, y, a, C, p, args...)
    y, grad
end
