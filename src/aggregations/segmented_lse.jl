# https://arxiv.org/abs/1511.05286
struct SegmentedLSE{T, U} <: AggregationFunction
    ρ::T
    C::U
end

Flux.@functor SegmentedLSE

SegmentedLSE(d::Int) = SegmentedLSE(randn(Float32, d), zeros(Float32, d))

r_map(ρ) = softplus.(ρ)
inv_r_map = (r) -> max.(r, 0) .+ log1p.(-exp.(-abs.(r)))

(m::SegmentedLSE)(x::MaybeMatrix, bags::AbstractBags, w=nothing) = segmented_lse_optim(x, m.C, r_map(m.ρ), bags)
function (m::SegmentedLSE)(x::AbstractMatrix, bags::AbstractBags, w::AggregationWeights, mask::AbstractVector)
    segmented_lse_optim(x .+ typemin(T) * mask', m.C, r_map(m.ρ), bags)
end

segmented_lse_optim(x::Missing, C::AbstractVector, r::AbstractVector, bags::AbstractBags) = segmented_lse_forw(x, C, bags)
function segmented_lse_optim(x::AbstractMatrix, C::AbstractVector, r::AbstractVector, bags::AbstractBags)
    y = (segmented_lse_forw(r .* x, C, bags)) ./ r
end

segmented_lse_forw(::Missing, C::AbstractVector, bags::AbstractBags) = repeat(C, 1, length(bags))
function segmented_lse_forw(x::AbstractMatrix, C::AbstractVector, bags::AbstractBags)
    y = zeros(eltype(x), length(C), length(bags))
    M = zeros(eltype(x), length(C))

    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                y[i, bi] = C[i]
            end
        else
            for i in eachindex(C)
                M[i] = x[i, first(b)]
            end
            for j in b[2:end]
                for i in eachindex(C)
                    M[i] = max.(M[i], x[i, j])
                end
            end
            for j in b
                for i in eachindex(C)
                    y[i, bi] += exp.(x[i, j] - M[i])
                end
            end
            for i in eachindex(C)
                y[i, bi] = log(y[i, bi]) - log(length(b)) + M[i]
            end
        end
    end
    y
end

function segmented_lse_back(Δ, x, C, bags)
    dx = zero(x)
    dC = zero(C)
    M = zeros(eltype(x), length(C))
    S = zeros(eltype(x), length(C))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                dC[i] += Δ[i, bi]
            end
        else
            for i in eachindex(C)
                M[i] = x[i, first(b)]
                S[i] = 0
            end
            for j in b[2:end]
                for i in eachindex(C)
                    M[i] = max(M[i], x[i, j])
                end
            end
            for j in b
                for i in eachindex(C)
                    S[i] += exp(x[i, j] - M[i])
                end
            end
            for j in b
                for i in eachindex(C)
                    dx[i, j] = Δ[i, bi] * exp(x[i, j] - M[i]) / S[i]
                end
            end
        end
    end
    dx, dC, nothing, nothing
end

function segmented_lse_back(Δ, C, bags)
    dC = zero(C)
    @inbounds for (bi, b) in enumerate(bags)
        for i in eachindex(C)
            dC[i] += Δ[i, bi]
        end
    end
    nothing, dC, nothing, nothing
end

@adjoint function segmented_lse_forw(x::AbstractMatrix, C::AbstractVector, bags::AbstractBags)
    y = segmented_lse_forw(x, C, bags)
    grad = Δ -> segmented_lse_back(Δ, x, C, bags)
    y, grad
end
