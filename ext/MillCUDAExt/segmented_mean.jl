# CUDA implementation of SegmentedMean

"""
    kernel_segmented_mean_forw!(o, x, ψ, indices, bags)

CUDA kernel for SegmentedMean forward pass without weights.
"""
function kernel_segmented_mean_forw!(o, x, ψ, indices, bags)
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
            @inbounds for j in bag
                acc += x[row, indices[j]]
                count += 1
            end
            @inbounds o[row, col] = acc / count
        end
    end
    return nothing
end

"""
    kernel_segmented_mean_forw_w!(o, x, ψ, w, indices, bags)

CUDA kernel for SegmentedMean forward pass with vector weights.
"""
function kernel_segmented_mean_forw_w!(o, x, ψ, w, indices, bags)
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
            @inbounds for j in bag
                idx_j = indices[j]
                acc += w[idx_j] * x[row, idx_j]
                wsum += w[idx_j]
            end
            @inbounds o[row, col] = acc / wsum
        end
    end
    return nothing
end

"""
    kernel_segmented_mean_forw_wm!(o, x, ψ, w, indices, bags)

CUDA kernel for SegmentedMean forward pass with matrix weights.
"""
function kernel_segmented_mean_forw_wm!(o, x, ψ, w, indices, bags)
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
            @inbounds for j in bag
                idx_j = indices[j]
                acc += w[row, idx_j] * x[row, idx_j]
                wsum += w[row, idx_j]
            end
            @inbounds o[row, col] = acc / wsum
        end
    end
    return nothing
end

"""
    kernel_segmented_mean_back!(dx, Δ, indices, bags)

CUDA kernel for SegmentedMean backward pass without weights.
"""
function kernel_segmented_mean_back!(dx, Δ, indices, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(Δ)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if !isempty(bag)
            bag_len = length(bag)
            @inbounds for j in bag
                CUDA.@atomic dx[row, indices[j]] += Δ[row, col] / bag_len
            end
        end
    end
    return nothing
end

"""
    kernel_segmented_mean_back_w!(dx, dw, Δ, x, y, w, indices, bags)

CUDA kernel for SegmentedMean backward pass with vector weights.
"""
function kernel_segmented_mean_back_w!(dx, dw, Δ, x, y, w, indices, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(Δ)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if !isempty(bag)
            wsum = zero(eltype(w))
            @inbounds for j in bag
                wsum += w[indices[j]]
            end
            @inbounds for j in bag
                idx_j = indices[j]
                CUDA.@atomic dx[row, idx_j] += w[idx_j] * Δ[row, col] / wsum
                CUDA.@atomic dw[idx_j] += Δ[row, col] * (x[row, idx_j] - y[row, col]) / wsum
            end
        end
    end
    return nothing
end

# Forward pass dispatches
function segmented_mean_forw(x::CuMatrix, ψ::CuVector, bags::CuCompressedBags, w::Nothing)
    o = CUDA.zeros(eltype(x), size(x, 1), length(bags.bags))
    n_elements = length(o)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_mean_forw!(o, x, ψ, bags.indices, bags.bags)
    end
    o
end

function segmented_mean_forw(x::CuMatrix, ψ::CuVector, bags::CuCompressedBags, w::CuVector)
    o = CUDA.zeros(eltype(x), size(x, 1), length(bags.bags))
    n_elements = length(o)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_mean_forw_w!(o, x, ψ, w, bags.indices, bags.bags)
    end
    o
end

function segmented_mean_forw(x::CuMatrix, ψ::CuVector, bags::CuCompressedBags, w::CuMatrix)
    o = CUDA.zeros(eltype(x), size(x, 1), length(bags.bags))
    n_elements = length(o)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_mean_forw_wm!(o, x, ψ, w, bags.indices, bags.bags)
    end
    o
end

# Missing data dispatch
segmented_mean_forw(::Missing, ψ::CuVector, bags::CuCompressedBags, w) = repeat(ψ, 1, length(bags.bags))

# Backward pass
function segmented_mean_back(Δ::CuMatrix, y, x::CuMatrix, ψ::CuVector, bags::CuCompressedBags, w::Nothing)
    dx = CUDA.zeros(eltype(x), size(x))
    dψ = CUDA.zeros(eltype(ψ), length(ψ))

    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_mean_back!(dx, Δ, bags.indices, bags.bags)
        @cuda threads=threads blocks=blocks kernel_sum_to_ψ!(dψ, Δ, bags.bags)
    end

    dx, dψ, ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent()
end

function segmented_mean_back(Δ::CuMatrix, y::CuMatrix, x::CuMatrix, ψ::CuVector, bags::CuCompressedBags, w::CuVector)
    dx = CUDA.zeros(eltype(x), size(x))
    dw = CUDA.zeros(eltype(w), length(w))
    dψ = CUDA.zeros(eltype(ψ), length(ψ))

    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_mean_back_w!(dx, dw, Δ, x, y, w, bags.indices, bags.bags)
        @cuda threads=threads blocks=blocks kernel_sum_to_ψ!(dψ, Δ, bags.bags)
    end

    dx, dψ, ChainRulesCore.NoTangent(), dw
end

function segmented_mean_back(Δ::CuMatrix, y, x::Missing, ψ::CuVector, bags::CuCompressedBags, w)
    dψ = CUDA.zeros(eltype(ψ), length(ψ))
    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_sum_to_ψ!(dψ, Δ, bags.bags)
    end
    ChainRulesCore.ZeroTangent(), dψ, ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent()
end

# ChainRulesCore rrule
function ChainRulesCore.rrule(::typeof(segmented_mean_forw), x::CuMatrix, ψ::CuVector, bags::CuCompressedBags, w)
    y = segmented_mean_forw(x, ψ, bags, w)
    function segmented_mean_pullback(Δ)
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        grads = segmented_mean_back(Δ_unthunked, y, x, ψ, bags, w)
        (ChainRulesCore.NoTangent(), grads...)
    end
    y, segmented_mean_pullback
end

function ChainRulesCore.rrule(::typeof(segmented_mean_forw), x::Missing, ψ::CuVector, bags::CuCompressedBags, w)
    y = segmented_mean_forw(x, ψ, bags, w)
    function segmented_mean_pullback(Δ)
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        grads = segmented_mean_back(Δ_unthunked, y, x, ψ, bags, w)
        (ChainRulesCore.NoTangent(), grads...)
    end
    y, segmented_mean_pullback
end
