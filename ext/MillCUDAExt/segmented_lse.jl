# CUDA implementation of SegmentedLSE (Log-Sum-Exp)

"""
    kernel_lse_precomp!(M, x, r, indices, bags)

CUDA kernel to precompute max(r * x) per bag for numerical stability in LSE.
"""
function kernel_lse_precomp!(M, x, r, indices, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(M)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if isempty(bag)
            @inbounds M[row, col] = zero(eltype(M))
        else
            @inbounds ri = r[row]
            max_val = typemin(eltype(M))
            @inbounds for j in bag
                val = ri * x[row, indices[j]]
                max_val = max(max_val, val)
            end
            @inbounds M[row, col] = max_val
        end
    end
    return nothing
end

"""
    kernel_segmented_lse_forw!(o, x, ψ, r, M, indices, bags)

CUDA kernel for SegmentedLSE forward pass.
Uses log-sum-exp trick for numerical stability.
"""
function kernel_segmented_lse_forw!(o, x, ψ, r, M, indices, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(o)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if isempty(bag)
            @inbounds o[row, col] = ψ[row]
        else
            @inbounds ri = r[row]
            @inbounds mi = M[row, col]
            count = 0
            acc = zero(eltype(o))
            @inbounds for j in bag
                acc += exp(ri * x[row, indices[j]] - mi)
                count += 1
            end
            @inbounds o[row, col] = (log(acc) - log(count) + mi) / ri
        end
    end
    return nothing
end

"""
    kernel_segmented_lse_back!(dx, dr, dψ, Δ, y, x, r, M, indices, bags)

CUDA kernel for SegmentedLSE backward pass.
"""
function kernel_segmented_lse_back!(dx, dr, dψ, Δ, y, x, r, M, indices, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(Δ)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if isempty(bag)
            @inbounds CUDA.@atomic dψ[row] += Δ[row, col]
        else
            @inbounds ri = r[row]
            @inbounds mi = M[row, col]
            @inbounds yi = y[row, col]
            @inbounds δ = Δ[row, col]

            # Compute sums for gradients
            s1 = zero(eltype(x))  # sum of exp
            s2 = zero(eltype(x))  # sum of x * exp

            @inbounds for j in bag
                idx_j = indices[j]
                e = exp(ri * x[row, idx_j] - mi)
                s1 += e
                s2 += x[row, idx_j] * e
            end

            # Gradient for x
            @inbounds for j in bag
                idx_j = indices[j]
                e = exp(ri * x[row, idx_j] - mi)
                CUDA.@atomic dx[row, idx_j] += δ * e / s1
            end

            # Gradient for r
            CUDA.@atomic dr[row] += δ * (s2 / s1 - yi) / ri
        end
    end
    return nothing
end

# Precompute M on GPU
function lse_precomp_gpu(x::CuMatrix, r::CuVector, bags::CuCompressedBags)
    M = CUDA.zeros(eltype(x), size(x, 1), length(bags.bags))
    n_elements = length(M)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_lse_precomp!(M, x, r, bags.indices, bags.bags)
    end
    M
end

# Forward pass with M storage for backward
function segmented_lse_forw_with_M(x::CuMatrix, ψ::CuVector, r::CuVector, bags::CuCompressedBags)
    M = lse_precomp_gpu(x, r, bags)
    o = CUDA.zeros(eltype(x), size(x, 1), length(bags.bags))
    n_elements = length(o)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_lse_forw!(o, x, ψ, r, M, bags.indices, bags.bags)
    end
    o, M
end

# Forward pass dispatch
function segmented_lse_forw(x::CuMatrix, ψ::CuVector, r::CuVector, bags::CuCompressedBags)
    segmented_lse_forw_with_M(x, ψ, r, bags)[1]
end

# Missing data dispatch
segmented_lse_forw(::Missing, ψ::CuVector, r::CuVector, bags::CuCompressedBags) = repeat(ψ, 1, length(bags.bags))

# Backward pass
function segmented_lse_back(Δ::CuMatrix, y::CuMatrix, x::CuMatrix, ψ::CuVector, r::CuVector,
                            bags::CuCompressedBags, M::CuMatrix)
    dx = CUDA.zeros(eltype(x), size(x))
    dr = CUDA.zeros(eltype(r), length(r))
    dψ = CUDA.zeros(eltype(ψ), length(ψ))

    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_lse_back!(dx, dr, dψ, Δ, y, x, r, M, bags.indices, bags.bags)
    end

    dx, dψ, dr, ChainRulesCore.NoTangent()
end

function segmented_lse_back(Δ::CuMatrix, x::Missing, ψ::CuVector, bags::CuCompressedBags)
    dψ = CUDA.zeros(eltype(ψ), length(ψ))
    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_sum_to_ψ!(dψ, Δ, bags.bags)
    end
    ChainRulesCore.ZeroTangent(), dψ, ChainRulesCore.ZeroTangent(), ChainRulesCore.NoTangent()
end

# ChainRulesCore rrule
function ChainRulesCore.rrule(::typeof(segmented_lse_forw), x::CuMatrix, ψ::CuVector, r::CuVector, bags::CuCompressedBags)
    y, M = segmented_lse_forw_with_M(x, ψ, r, bags)
    function segmented_lse_pullback(Δ)
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        grads = segmented_lse_back(Δ_unthunked, y, x, ψ, r, bags, M)
        (ChainRulesCore.NoTangent(), grads...)
    end
    y, segmented_lse_pullback
end

function ChainRulesCore.rrule(::typeof(segmented_lse_forw), x::Missing, ψ::CuVector, r::CuVector, bags::CuCompressedBags)
    y = segmented_lse_forw(x, ψ, r, bags)
    function segmented_lse_pullback(Δ)
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        grads = segmented_lse_back(Δ_unthunked, x, ψ, bags)
        (ChainRulesCore.NoTangent(), grads...)
    end
    y, segmented_lse_pullback
end
