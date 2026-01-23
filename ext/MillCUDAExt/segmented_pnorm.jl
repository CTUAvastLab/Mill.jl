# CUDA implementation of SegmentedPNorm

"""
    kernel_pnorm_precomp!(M, x, indices, bags)

CUDA kernel to precompute max absolute values per bag for numerical stability.
"""
function kernel_pnorm_precomp!(M, x, indices, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(M)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if isempty(bag)
            @inbounds M[row, col] = one(eltype(M))
        else
            max_val = one(eltype(M))
            @inbounds for j in bag
                max_val = max(max_val, abs(x[row, indices[j]]))
            end
            @inbounds M[row, col] = max_val
        end
    end
    return nothing
end

"""
    kernel_segmented_pnorm_forw!(o, a, ψ, p, M, indices, bags)

CUDA kernel for SegmentedPNorm forward pass without weights.
a = x - c (already shifted by center)
"""
function kernel_segmented_pnorm_forw!(o, a, ψ, p, M, indices, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(o)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if isempty(bag)
            @inbounds o[row, col] = ψ[row]
        else
            acc = zero(eltype(o))
            count = 0
            @inbounds m = M[row, col]
            @inbounds pi = p[row]
            @inbounds for j in bag
                val = abs(a[row, indices[j]] / m)
                acc += val ^ pi
                count += 1
            end
            @inbounds o[row, col] = m * (acc / count) ^ (one(eltype(o)) / pi)
        end
    end
    return nothing
end

"""
    kernel_segmented_pnorm_forw_w!(o, a, ψ, p, w, M, indices, bags)

CUDA kernel for SegmentedPNorm forward pass with vector weights.
"""
function kernel_segmented_pnorm_forw_w!(o, a, ψ, p, w, M, indices, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(o)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if isempty(bag)
            @inbounds o[row, col] = ψ[row]
        else
            acc = zero(eltype(o))
            wsum = zero(eltype(w))
            @inbounds m = M[row, col]
            @inbounds pi = p[row]
            @inbounds for j in bag
                idx_j = indices[j]
                val = abs(a[row, idx_j] / m)
                acc += w[idx_j] * val ^ pi
                wsum += w[idx_j]
            end
            @inbounds o[row, col] = m * (acc / wsum) ^ (one(eltype(o)) / pi)
        end
    end
    return nothing
end

"""
    kernel_segmented_pnorm_back!(da, dp, dψ, Δ, y, a, ψ, p, M, indices, bags)

CUDA kernel for SegmentedPNorm backward pass without weights.
"""
function kernel_segmented_pnorm_back!(da, dp, dψ, Δ, y, a, ψ, p, M, indices, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(Δ)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if isempty(bag)
            @inbounds CUDA.@atomic dψ[row] += Δ[row, col]
        else
            @inbounds pi = p[row]
            @inbounds m = M[row, col]
            @inbounds yi = y[row, col]
            @inbounds δ = Δ[row, col]

            # Compute sums for dp gradient
            dps1 = zero(eltype(a))
            dps2 = zero(eltype(a))
            count = 0

            @inbounds for j in bag
                idx_j = indices[j]
                ab = abs(a[row, idx_j])
                ww = (ab / m) ^ pi
                dps1 += ww * log(ab + eps(eltype(a)))
                dps2 += ww
                count += 1
            end

            # Gradient for a
            @inbounds for j in bag
                idx_j = indices[j]
                ab = abs(a[row, idx_j])
                da_grad = δ * sign(a[row, idx_j]) / count * (ab / yi) ^ (pi - one(eltype(a)))
                CUDA.@atomic da[row, idx_j] += da_grad
            end

            # Gradient for p
            t = yi / pi
            t *= dps1 / dps2 - (pi * log(m) + log(dps2) - log(count)) / pi
            CUDA.@atomic dp[row] += δ * t
        end
    end
    return nothing
end

# Precompute M on GPU
function pnorm_precomp_gpu(a::CuMatrix, bags::CuCompressedBags)
    M = CUDA.ones(eltype(a), size(a, 1), length(bags.bags))
    n_elements = length(M)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_pnorm_precomp!(M, a, bags.indices, bags.bags)
    end
    M
end

# Forward pass dispatch
function segmented_pnorm_forw(a::CuMatrix, ψ::CuVector, p::CuVector, bags::CuCompressedBags, w::Nothing)
    M = pnorm_precomp_gpu(a, bags)
    o = CUDA.zeros(eltype(a), size(a, 1), length(bags.bags))
    n_elements = length(o)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_pnorm_forw!(o, a, ψ, p, M, bags.indices, bags.bags)
    end
    o, M
end

function segmented_pnorm_forw(a::CuMatrix, ψ::CuVector, p::CuVector, bags::CuCompressedBags, w::CuVector)
    M = pnorm_precomp_gpu(a, bags)
    o = CUDA.zeros(eltype(a), size(a, 1), length(bags.bags))
    n_elements = length(o)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_pnorm_forw_w!(o, a, ψ, p, w, M, bags.indices, bags.bags)
    end
    o, M
end

# Wrapper that discards M for simple forward
function Mill.segmented_pnorm_forw(a::CuMatrix, ψ::CuVector, p::CuVector, bags::CuCompressedBags, w)
    segmented_pnorm_forw(a, ψ, p, bags, w)[1]
end

# Missing data dispatch
segmented_pnorm_forw(::Missing, ψ::CuVector, p, bags::CuCompressedBags, w) = repeat(ψ, 1, length(bags.bags))

# Backward pass
function segmented_pnorm_back(Δ::CuMatrix, y::CuMatrix, a::CuMatrix, ψ::CuVector, p::CuVector,
                              bags::CuCompressedBags, w::Nothing, M::CuMatrix)
    da = CUDA.zeros(eltype(a), size(a))
    dp = CUDA.zeros(eltype(p), length(p))
    dψ = CUDA.zeros(eltype(ψ), length(ψ))

    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_pnorm_back!(da, dp, dψ, Δ, y, a, ψ, p, M, bags.indices, bags.bags)
    end

    da, dψ, dp, ChainRulesCore.NoTangent(), ChainRulesCore.@not_implemented("Weight gradients not implemented for GPU PNorm")
end

function segmented_pnorm_back(Δ::CuMatrix, y, a::Missing, ψ::CuVector, bags::CuCompressedBags)
    dψ = CUDA.zeros(eltype(ψ), length(ψ))
    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_sum_to_ψ!(dψ, Δ, bags.bags)
    end
    ChainRulesCore.ZeroTangent(), dψ, ChainRulesCore.ZeroTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.@not_implemented("Weight gradients not implemented for GPU PNorm")
end

# ChainRulesCore rrule
function ChainRulesCore.rrule(::typeof(Mill.segmented_pnorm_forw), a::CuMatrix, ψ::CuVector, p::CuVector, bags::CuCompressedBags, w)
    y, M = segmented_pnorm_forw(a, ψ, p, bags, w)
    function segmented_pnorm_pullback(Δ)
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        grads = segmented_pnorm_back(Δ_unthunked, y, a, ψ, p, bags, w, M)
        (ChainRulesCore.NoTangent(), grads...)
    end
    y, segmented_pnorm_pullback
end

function ChainRulesCore.rrule(::typeof(Mill.segmented_pnorm_forw), a::Missing, ψ::CuVector, p, bags::CuCompressedBags, w)
    y = segmented_pnorm_forw(a, ψ, p, bags, w)
    function segmented_pnorm_pullback(Δ)
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        grads = segmented_pnorm_back(Δ_unthunked, y, a, ψ, bags)
        (ChainRulesCore.NoTangent(), grads...)
    end
    y, segmented_pnorm_pullback
end
