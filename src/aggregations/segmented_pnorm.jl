# https://arxiv.org/pdf/1311.1780.pdf
struct SegmentedPNorm{T, V <: AbstractVector{T}} <: AggregationOperator{T}
    ρ::V
    c::V
    ψ::V
end

Flux.@functor SegmentedPNorm

_SegmentedPNorm(d::Int) = SegmentedPNorm(randn(Float32, d), randn(Float32, d), zeros(Float32, d))

Flux.@forward SegmentedPNorm.ψ Base.getindex, Base.length, Base.size, Base.firstindex, Base.lastindex,
        Base.first, Base.last, Base.iterate, Base.eltype

Base.vcat(as::SegmentedPNorm...) = reduce(vcat, as |> collect)
function Base.reduce(::typeof(vcat), as::Vector{<:SegmentedPNorm})
    SegmentedPNorm(reduce(vcat, [a.ρ for a in as]),
                   reduce(vcat, [a.c for a in as]),
                   reduce(vcat, [a.ψ for a in as]))
end

p_map(ρ::T) where T = one(T) + softplus(ρ)
p_map(ρ::AbstractArray) = p_map.(ρ)
inv_p_map(p::T) where T = relu(p - one(T)) + log1p(-exp(-abs(p - one(T))))
inv_p_map(ρ::AbstractArray) = inv_p_map.(ρ)

(m::SegmentedPNorm)(x::Missing, bags::AbstractBags, w=nothing) = segmented_pnorm_forw(x, m.ψ, nothing, bags, w)
(m::SegmentedPNorm)(x::AbstractMatrix, bags::AbstractBags, w=nothing) = segmented_pnorm_forw(x .- m.c, m.ψ, p_map(m.ρ), bags, w)
function (m::SegmentedPNorm)(x::AbstractMatrix, bags::AbstractBags, w::Optional{AbstractVecOrMat}, mask::AbstractVector)
    segmented_pnorm_forw((x.-m.c) .* mask', m.ψ, p_map(m.ρ), bags, w)
end

function _pnorm_precomp(x::AbstractMatrix, bags)
    M = ones(eltype(x), size(x, 1), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if !isempty(b)
            for j in b
                for i in 1:size(x, 1)
                    M[i, bi] = max(M[i, bi], abs(x[i, j]))
                end
            end
        end
    end
    M
end

function _segmented_pnorm_norm(a::AbstractMatrix, ψ::AbstractVector, p::AbstractVector,
                               bags::AbstractBags, w, M)
    isnothing(w) || @assert all(w .> 0)
    y = zeros(eltype(a), size(a, 1), length(bags))
    M = _pnorm_precomp(a, bags)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                y[i, bi] = ψ[i]
            end
        else
            ws = bagnorm(w, b)
            for j in b
                for i in 1:size(a, 1)
                    y[i, bi] += weight(w, i, j, eltype(a)) * abs(a[i, j]/M[i, bi])^p[i]
                end
            end
            for i in 1:size(a, 1)
                y[i, bi] = M[i, bi] * (y[i, bi] / weightsum(ws, i))^(one(eltype(a))/p[i])
            end
        end
    end
    y
end

segmented_pnorm_forw(::Missing, ψ::AbstractVector, p, bags::AbstractBags, w) = repeat(ψ, 1, length(bags))
function segmented_pnorm_forw(a::Maybe{AbstractMatrix}, ψ::AbstractVector, p::AbstractVector, bags::AbstractBags, w) 
    M = _pnorm_precomp(a, bags)
    _segmented_pnorm_norm(a, ψ, p, bags, w, M)
end

function segmented_pnorm_back(Δ, y, a, ψ, p, bags, w, M)
    da = similar(a)
    dp = zero(p)
    dps1 = zero(p)
    dps2 = zero(p)
    dψ = zero(ψ)
    dw = isnothing(w) ? nothing : zero(w)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                dψ[i] += Δ[i, bi]
            end
        else
            ws = bagnorm(w, b)
            dps1 .= zero(eltype(p))
            dps2 .= zero(eltype(p))
            for j in b
                for i in 1:size(a, 1)
                    ab = abs(a[i, j])
                    da[i, j] = Δ[i, bi] * weight(w, i, j, eltype(p)) * sign(a[i, j]) / weightsum(ws, i) 
                    da[i, j] *= (ab / y[i, bi]) ^ (p[i] - one(eltype(p)))
                    ww = weight(w, i, j, eltype(p)) * (ab / M[i, bi]) ^ p[i]
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
    da, dψ, dp, DoesNotExist(), dw
end

function segmented_pnorm_back(Δ, y, ψ, bags) 
    dψ = zero(ψ)
    @inbounds for (bi, b) in enumerate(bags)
        for i in eachindex(ψ)
            dψ[i] += Δ[i, bi]
        end
    end
    Zero(), dψ, Zero(), DoesNotExist(), Zero()
end

∇dw_segmented_pnorm!(dw::Zero, Δ, a, y, w::Nothing, ws, i, j, bi) = error("Not implemented yet!")
function ∇dw_segmented_pnorm!(dw::AbstractVector, Δ, a, y, w::AbstractVector, ws, i, j, bi) 
    error("Not implemented yet!")
end
function ∇dw_segmented_pnorm!(dw::AbstractMatrix, Δ, a, y, w::AbstractMatrix, ws, i, j, bi)
    error("Not implemented yet!")
end

function rrule(::typeof(segmented_pnorm_forw), a::AbstractMatrix, ψ, p, bags, w)
    M = _pnorm_precomp(a, bags)
    y = _segmented_pnorm_norm(a, ψ, p, bags, w, M)
    grad = Δ -> (NO_FIELDS, segmented_pnorm_back(Δ, y, a, ψ, p, bags, w, M)...)
    y, grad
end

function rrule(::typeof(segmented_pnorm_forw), a::Missing, ψ, p, bags, w)
    y = segmented_pnorm_forw(a, ψ, p, bags, w)
    grad = Δ -> (NO_FIELDS, segmented_pnorm_back(Δ, y, ψ, bags)...)
    y, grad
end
