# CUDA implementation of SegmentedMax

"""
    kernel_segmented_max_forw!(o, maxI, x, ψ, indices, bags)

CUDA kernel for SegmentedMax forward pass.
Also stores the index of the maximum element for backpropagation.
"""
function kernel_segmented_max_forw!(o, maxI, x, ψ, indices, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(o)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if isempty(bag)
            @inbounds o[row, col] = ψ[row]
            @inbounds maxI[row, col] = 0
        else
            max_val = typemin(eltype(o))
            max_idx = Int32(0)
            @inbounds for j in bag
                idx_j = indices[j]
                val = x[row, idx_j]
                if val > max_val
                    max_val = val
                    max_idx = idx_j
                end
            end
            @inbounds o[row, col] = max_val
            @inbounds maxI[row, col] = max_idx
        end
    end
    return nothing
end

"""
    kernel_segmented_max_back!(dx, Δ, maxI, bags)

CUDA kernel for SegmentedMax backward pass.
Uses the stored max indices to route gradients.
"""
function kernel_segmented_max_back!(dx, Δ, maxI, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(Δ)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if !isempty(bag)
            @inbounds max_idx = maxI[row, col]
            if max_idx > 0
                @inbounds CUDA.@atomic dx[row, max_idx] += Δ[row, col]
            end
        end
    end
    return nothing
end

# Forward pass with max index storage for backward
function segmented_max_forw_with_maxI(x::CuMatrix, ψ::CuVector, bags::CuCompressedBags)
    o = CUDA.zeros(eltype(x), size(x, 1), length(bags.bags))
    maxI = CUDA.zeros(Int32, size(x, 1), length(bags.bags))

    n_elements = length(o)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_max_forw!(o, maxI, x, ψ, bags.indices, bags.bags)
    end
    o, maxI
end

# Forward pass dispatch
function segmented_max_forw(x::CuMatrix, ψ::CuVector, bags::CuCompressedBags)
    segmented_max_forw_with_maxI(x, ψ, bags)[1]
end

# Missing data dispatch
segmented_max_forw(::Missing, ψ::CuVector, bags::CuCompressedBags) = repeat(ψ, 1, length(bags.bags))

# Backward pass
function segmented_max_back(Δ::CuMatrix, maxI::CuMatrix{Int32}, y, x::CuMatrix, ψ::CuVector, bags::CuCompressedBags)
    dx = CUDA.zeros(eltype(x), size(x))
    dψ = CUDA.zeros(eltype(ψ), length(ψ))

    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_max_back!(dx, Δ, maxI, bags.bags)
        @cuda threads=threads blocks=blocks kernel_sum_to_ψ!(dψ, Δ, bags.bags)
    end

    dx, dψ, ChainRulesCore.NoTangent()
end

function segmented_max_back(Δ::CuMatrix, y, x::Missing, ψ::CuVector, bags::CuCompressedBags)
    dψ = CUDA.zeros(eltype(ψ), length(ψ))
    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_sum_to_ψ!(dψ, Δ, bags.bags)
    end
    ChainRulesCore.ZeroTangent(), dψ, ChainRulesCore.NoTangent()
end

# ChainRulesCore rrule
function ChainRulesCore.rrule(::typeof(segmented_max_forw), x::CuMatrix, ψ::CuVector, bags::CuCompressedBags)
    y, maxI = segmented_max_forw_with_maxI(x, ψ, bags)
    function segmented_max_pullback(Δ)
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        grads = segmented_max_back(Δ_unthunked, maxI, y, x, ψ, bags)
        (ChainRulesCore.NoTangent(), grads...)
    end
    y, segmented_max_pullback
end

function ChainRulesCore.rrule(::typeof(segmented_max_forw), x::Missing, ψ::CuVector, bags::CuCompressedBags)
    y = segmented_max_forw(x, ψ, bags)
    function segmented_max_pullback(Δ)
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        grads = segmented_max_back(Δ_unthunked, y, x, ψ, bags)
        (ChainRulesCore.NoTangent(), grads...)
    end
    y, segmented_max_pullback
end
