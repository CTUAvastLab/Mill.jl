# https://arxiv.org/pdf/1311.1780.pdf
struct PNorm{T}
    ρ::T
    c::T
end

PNorm(d::Int) = PNorm(param(randn(d)), param(randn(d)))
Flux.treelike(PNorm)

p_map(ρ) = 1 .+ log.(1 .+ exp.(ρ))
inv_p_map(p) = log.(exp.(p-1) .- 1)

function segmented_pnorm(x::Matrix, p::Vector, c::Vector, bags::Bags)
    o = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (j, b) in enumerate(bags)
        for bi in b
            for i in 1:size(x, 1)
                o[i, j] += abs(x[i, bi] - c[i]) ^ p[i] / length(b)
            end
        end
        o[:, j] .^= 1 ./ p
    end
    o
end

function segmented_pnorm(x::Matrix, p::Vector, c::Vector, bags::Bags, w::Vector)
    @assert all(w .> 0)
    o = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(w[b])
        for bi in b
            for i in 1:size(x, 1)
                o[i, j] += w[bi] * abs(x[i, bi] - c[i]) ^ p[i] / ws
            end
        end
        o[:, j] .^= 1 ./ p
    end
    o
end

segmented_pnorm_back(x::TrackedArray, p::Vector, ρ::Vector, c::Vector, bags::Bags, n::Matrix) = Δ -> begin
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    dx = zeros(x)
    @inbounds for (j, b) in enumerate(bags)
        for bi in b
            for i in 1:size(x,1)
                dx[i, bi] = Δ[i, j] * sign(x[i, bi] - c[i])
                dx[i, bi] /= length(b)
                dx[i, bi] *= (abs(x[i, bi] - c[i]) / n[i, j]) ^ (p[i] - 1)
            end
        end
    end
    dx, nothing, nothing , nothing
end

segmented_pnorm_back(x::TrackedArray, p::Vector, ρ::Vector, c::Vector, bags::Bags, w::Vector, n::Matrix) = Δ -> begin
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    dx = zeros(x)
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(w[b])
        for bi in b
            for i in 1:size(x,1)
                dx[i, bi] = Δ[i, j] * w[bi] * sign(x[i, bi] - c[i])
                dx[i, bi] /= ws
                dx[i, bi] *= (abs(x[i, bi] - c[i]) / n[i, j]) ^ (p[i] - 1)
            end
        end
    end
    dx, nothing, nothing , nothing, nothing
end

segmented_pnorm_back(x::TrackedArray, p::TrackedVector, ρ::TrackedVector, c::TrackedVector, bags::Bags, n::Matrix) = Δ -> begin
    x = Flux.data(x)
    p = Flux.data(p)
    ρ = Flux.data(ρ)
    c = Flux.data(c)
    Δ = Flux.data(Δ)
    dx = zeros(x)
    dp = zeros(p)
    dps1 = zeros(p)
    dps2 = zeros(p)
    dc = zeros(c)
    dcs = zeros(c)
    @inbounds for (j, b) in enumerate(bags)
        dcs .= 0
        dps1 .= 0
        dps2 .= 0
        for bi in b
            for i in 1:size(x,1)
                ab = abs(x[i, bi] - c[i])
                sig = sign(x[i, bi] - c[i])
                dx[i, bi] = Δ[i, j] * sig
                dx[i, bi] /= length(b)
                dx[i, bi] *= (ab / n[i, j]) ^ (p[i] - 1)
                dps1[i] += ab ^ p[i] * log(ab)
                dps2[i] += ab ^ p[i]
                dcs[i] -= sig * (ab ^ (p[i] - 1))
            end
        end
        t = n[:, j] ./ p .* (dps1 ./ dps2 .- (log.(dps2) .- log(max(1, length(b)))) ./ p)
        dp .+= Δ[:, j] .* t
        dcs ./= max(1, length(b))
        dcs .*= n[:, j] .^ (1 .- p)
        dc .+= Δ[:, j] .* dcs
    end
    dρ = dp .* σ.(ρ)
    dx, dρ, dc, nothing
end

segmented_pnorm_back(x::TrackedArray, p::TrackedVector, ρ::TrackedVector, c::TrackedVector, bags::Bags, w::Vector, n::Matrix) = Δ -> begin
    x = Flux.data(x)
    p = Flux.data(p)
    ρ = Flux.data(ρ)
    c = Flux.data(c)
    Δ = Flux.data(Δ)
    dx = zeros(x)
    dp = zeros(p)
    dps1 = zeros(p)
    dps2 = zeros(p)
    dc = zeros(c)
    dcs = zeros(c)
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(w[b])
        dcs .= 0
        dps1 .= 0
        dps2 .= 0
        for bi in b
            for i in 1:size(x,1)
                ab = abs(x[i, bi] - c[i])
                sig = sign(x[i, bi] - c[i])
                dx[i, bi] = Δ[i, j] * w[bi] * sig
                dx[i, bi] /= ws
                dx[i, bi] *= (ab / n[i, j]) ^ (p[i] - 1)
                dps1[i] += w[bi] * ab ^ p[i] * log(ab)
                dps2[i] += w[bi] * ab ^ p[i]
                dcs[i] -= w[bi] * sig * (ab ^ (p[i] - 1))
            end
        end
        t = n[:, j] ./ p .* (dps1 ./ dps2 .- (log.(dps2) .- log(ws)) ./ p)
        dp .+= Δ[:, j] .* t
        dcs ./= ws
        dcs .*= n[:, j] .^ (1 .- p)
        dc .+= Δ[:, j] .* dcs
    end
    dρ = dp .* σ.(ρ)
    dx, dρ, dc, nothing, nothing
end

segmented_pnorm_back(x::Matrix, p::TrackedVector, ρ::TrackedVector, c::TrackedVector, bags::Bags, n::Matrix) = Δ -> begin
    p = Flux.data(p)
    ρ = Flux.data(ρ)
    c = Flux.data(c)
    Δ = Flux.data(Δ)
    dp = zeros(p)
    dps1 = zeros(p)
    dps2 = zeros(p)
    dc = zeros(c)
    dcs = zeros(c)
    @inbounds for (j, b) in enumerate(bags)
        dcs .= 0
        dps1 .= 0
        dps2 .= 0
        for bi in b
            for i in 1:size(x,1)
                ab = abs(x[i, bi] - c[i])
                sig = sign(x[i, bi] - c[i])
                dps1[i] +=  ab ^ p[i] * log(ab)
                dps2[i] +=  ab ^ p[i]
                dcs[i] -= sig * (ab ^ (p[i] - 1))
            end
        end
        t = n[:, j] ./ p .* (dps1 ./ dps2 .- (log.(dps2) .- log(max(1, length(b)))) ./ p)
        dp .+= Δ[:, j] .* t
        dcs ./= max(1, length(b))
        dcs .*= n[:, j] .^ (1 .- p)
        dc .+= Δ[:, j] .* dcs
    end
    dρ = dp .* σ.(ρ)
    nothing, dρ, dc, nothing
end

segmented_pnorm_back(x::Matrix, p::TrackedVector, ρ::TrackedVector, c::TrackedVector, bags::Bags, w::Vector, n::Matrix) = Δ -> begin
    p = Flux.data(p)
    ρ = Flux.data(ρ)
    c = Flux.data(c)
    Δ = Flux.data(Δ)
    dp = zeros(p)
    dps1 = zeros(p)
    dps2 = zeros(p)
    dc = zeros(c)
    dcs = zeros(c)
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(w[b])
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
            end
        end
        t = n[:, j] ./ p .* (dps1 ./ dps2 .- (log.(dps2) .- log(ws)) ./ p)
        dp .+= Δ[:, j] .* t
        dcs ./= ws
        dcs .*= n[:, j] .^ (1 .- p)
        dc .+= Δ[:, j] .* dcs
    end
    dρ = dp .* σ.(ρ)
    nothing, dρ, dc, nothing, nothing
end

(n::PNorm)(x, args...) = segmented_pnorm(x, p_map(Flux.data(n.ρ)), Flux.data(n.c), args...)
(n::PNorm)(x::ArrayNode, args...) = mapdata(x -> n(x, args...), x)
(n::PNorm{<:TrackedVector})(x::ArrayNode, args...) = mapdata(x -> n(x, args...), x)

# both x and (ρ, c) can be params
(n::PNorm{<:AbstractVector})(x::TrackedArray, args...) = _pnorm_grad(x, n.ρ, n.c, args...)
(n::PNorm{<:TrackedVector})(x, args...) = _pnorm_grad(x, n.ρ, n.c, args...)
(n::PNorm{<:TrackedVector})(x::TrackedArray, args...) = _pnorm_grad(x, n.ρ, n.c, args...)

_pnorm_grad(x, ρ, c, args...) = Flux.Tracker.track(_pnorm_grad, x, ρ, c, args...)

Flux.Tracker.@grad function _pnorm_grad(x, ρ, c, args...)
    n = segmented_pnorm(Flux.data(x), p_map(Flux.data(ρ)), Flux.data(c), Flux.data.(args)...)
    grad = segmented_pnorm_back(x, p_map(ρ), ρ, c, args..., n)
    n, grad
end
