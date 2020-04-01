# https://arxiv.org/abs/1511.05286
struct SegmentedLSE{T, U} <: AggregationFunction
    ρ::T
    C::U
end

Flux.@functor SegmentedLSE

SegmentedLSE(d::Int) = SegmentedLSE(randn(Float32, d), zeros(Float32, d))

r_map(ρ) = softplus.(ρ)
inv_r_map = (r) -> log.(exp.(r) .- 1)

(m::SegmentedLSE)(x::MaybeMatrix, bags::AbstractBags, w=nothing) = segmented_lse_optim(x, m.C, r_map(m.ρ), bags)
function (m::SegmentedLSE)(x::AbstractMatrix, bags::AbstractBags, w::AggregationWeights, mask::AbstractVector)
    segmented_lse_optim(x .+ typemin(T) * mask', m.C, r_map(m.ρ), bags)
end

segmented_lse_optim(x::Missing, C::AbstractVector, p::AbstractVector, bags::AbstractBags) = segmented_lse_forw(x, C, bags)
function segmented_lse_optim(x::AbstractMatrix, C::AbstractVector, p::AbstractVector, bags::AbstractBags)
    a = p .* x
    m = maximum(a, dims=2)
    y = (m .+ segmented_lse_forw(a .- m, C, bags)) ./ p
end

segmented_lse_forw(::Missing, C::AbstractVector, bags::AbstractBags) = repeat(C, 1, length(bags))
function segmented_lse_forw(a::AbstractMatrix, C::AbstractVector, bags::AbstractBags) 
    y = zeros(eltype(a), size(a, 1), length(bags))
    e = exp.(a)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                y[i, bi] = C[i]
            end
        else
            for j in b
                for i in 1:size(a, 1)
                    y[i, bi] += e[i, j]
                end
            end
            lb = log(max(1, length(b)))
            for i in 1:size(a, 1)
                y[i, bi] = log(y[i, bi]) - lb
            end
        end
    end
    y
end

function segmented_lse_back(Δ, y, a, C, bags)
    da = zero(a)
    s = zero(C)
    dC = zero(C)
    e = exp.(a)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                dC[i] += Δ[i, bi]
            end
        else
            s .= 0
            for j in b
                for i in 1:size(a, 1)
                    da[i, j] = Δ[i, bi] * e[i, j]
                    s[i] += e[i, j]
                end
            end
            da[:, b] ./= s
        end
    end
    da, dC, nothing, nothing
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

@adjoint function segmented_lse_forw(a::AbstractMatrix, C::AbstractVector, bags::AbstractBags)
    y = segmented_lse_forw(a, C, bags)
    grad = Δ -> segmented_lse_back(Δ, y, a, C, bags)
    y, grad
end

@adjoint function segmented_lse_forw(a::Missing, C::AbstractVector, bags::AbstractBags)
    y = segmented_lse_forw(a, C, bags)
    grad = Δ -> segmented_lse_back(Δ, C, bags)
    y, grad
end
