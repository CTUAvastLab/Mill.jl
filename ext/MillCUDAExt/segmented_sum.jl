# CUDA implementation of SegmentedSum

"""
    kernel_segmented_sum_forw!(o, x, ψ, indices, bags)

CUDA kernel for SegmentedSum forward pass without weights.
Each thread computes one element (row, bag) of the output.
"""
function kernel_segmented_sum_forw!(o, x, ψ, indices, bags)
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
            @inbounds for j in bag
                acc += x[row, indices[j]]
            end
            @inbounds o[row, col] = acc
        end
    end
    return nothing
end

"""
    kernel_segmented_sum_forw_w!(o, x, ψ, w, indices, bags)

CUDA kernel for SegmentedSum forward pass with vector weights.
"""
function kernel_segmented_sum_forw_w!(o, x, ψ, w, indices, bags)
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
            @inbounds for j in bag
                acc += w[indices[j]] * x[row, indices[j]]
            end
            @inbounds o[row, col] = acc
        end
    end
    return nothing
end

"""
    kernel_segmented_sum_forw_wm!(o, x, ψ, w, indices, bags)

CUDA kernel for SegmentedSum forward pass with matrix weights.
"""
function kernel_segmented_sum_forw_wm!(o, x, ψ, w, indices, bags)
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
            @inbounds for j in bag
                acc += w[row, indices[j]] * x[row, indices[j]]
            end
            @inbounds o[row, col] = acc
        end
    end
    return nothing
end

"""
    kernel_segmented_sum_back!(dx, Δ, indices, bags)

CUDA kernel for SegmentedSum backward pass without weights.
Accumulates gradients using atomic operations.
"""
function kernel_segmented_sum_back!(dx, Δ, indices, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(Δ)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if !isempty(bag)
            @inbounds for j in bag
                CUDA.@atomic dx[row, indices[j]] += Δ[row, col]
            end
        end
    end
    return nothing
end

"""
    kernel_segmented_sum_back_w!(dx, dw, Δ, x, w, indices, bags)

CUDA kernel for SegmentedSum backward pass with vector weights.
"""
function kernel_segmented_sum_back_w!(dx, dw, Δ, x, w, indices, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(Δ)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if !isempty(bag)
            @inbounds for j in bag
                idx_j = indices[j]
                CUDA.@atomic dx[row, idx_j] += w[idx_j] * Δ[row, col]
                CUDA.@atomic dw[idx_j] += Δ[row, col] * x[row, idx_j]
            end
        end
    end
    return nothing
end

# Forward pass dispatches
function segmented_sum_forw(x::CuMatrix, ψ::CuVector, bags::CuCompressedBags, w::Nothing)
    o = CUDA.zeros(eltype(x), size(x, 1), length(bags.bags))
    n_elements = length(o)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_sum_forw!(o, x, ψ, bags.indices, bags.bags)
    end
    o
end

function segmented_sum_forw(x::CuMatrix, ψ::CuVector, bags::CuCompressedBags, w::CuVector)
    o = CUDA.zeros(eltype(x), size(x, 1), length(bags.bags))
    n_elements = length(o)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_sum_forw_w!(o, x, ψ, w, bags.indices, bags.bags)
    end
    o
end

function segmented_sum_forw(x::CuMatrix, ψ::CuVector, bags::CuCompressedBags, w::CuMatrix)
    o = CUDA.zeros(eltype(x), size(x, 1), length(bags.bags))
    n_elements = length(o)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_sum_forw_wm!(o, x, ψ, w, bags.indices, bags.bags)
    end
    o
end

# Missing data dispatch
segmented_sum_forw(::Missing, ψ::CuVector, bags::CuCompressedBags, w) = repeat(ψ, 1, length(bags.bags))

# Backward pass
function segmented_sum_back(Δ::CuMatrix, y, x::CuMatrix, ψ::CuVector, bags::CuCompressedBags, w::Nothing)
    dx = CUDA.zeros(eltype(x), size(x))
    dψ = CUDA.zeros(eltype(ψ), length(ψ))

    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_sum_back!(dx, Δ, bags.indices, bags.bags)
        @cuda threads=threads blocks=blocks kernel_sum_to_ψ!(dψ, Δ, bags.bags)
    end

    dx, dψ, ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent()
end

function segmented_sum_back(Δ::CuMatrix, y, x::CuMatrix, ψ::CuVector, bags::CuCompressedBags, w::CuVector)
    dx = CUDA.zeros(eltype(x), size(x))
    dw = CUDA.zeros(eltype(w), length(w))
    dψ = CUDA.zeros(eltype(ψ), length(ψ))

    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_sum_back_w!(dx, dw, Δ, x, w, bags.indices, bags.bags)
        @cuda threads=threads blocks=blocks kernel_sum_to_ψ!(dψ, Δ, bags.bags)
    end

    dx, dψ, ChainRulesCore.NoTangent(), dw
end

function segmented_sum_back(Δ::CuMatrix, y, x::Missing, ψ::CuVector, bags::CuCompressedBags, w)
    dψ = CUDA.zeros(eltype(ψ), length(ψ))
    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_sum_to_ψ!(dψ, Δ, bags.bags)
    end
    ChainRulesCore.ZeroTangent(), dψ, ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent()
end

# ChainRulesCore rrule
function ChainRulesCore.rrule(::typeof(segmented_sum_forw), x::CuMatrix, ψ::CuVector, bags::CuCompressedBags, w)
    y = segmented_sum_forw(x, ψ, bags, w)
    function segmented_sum_pullback(Δ)
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        grads = segmented_sum_back(Δ_unthunked, y, x, ψ, bags, w)
        (ChainRulesCore.NoTangent(), grads...)
    end
    y, segmented_sum_pullback
end

function ChainRulesCore.rrule(::typeof(segmented_sum_forw), x::Missing, ψ::CuVector, bags::CuCompressedBags, w)
    y = segmented_sum_forw(x, ψ, bags, w)
    function segmented_sum_pullback(Δ)
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        grads = segmented_sum_back(Δ_unthunked, y, x, ψ, bags, w)
        (ChainRulesCore.NoTangent(), grads...)
    end
    y, segmented_sum_pullback
end

# CuAlignedBags kernels: bags[col] is a direct UnitRange of instance indices (no indirection)

function kernel_segmented_sum_forw_aligned!(o, x, ψ, bags)
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
            @inbounds for j in bag
                acc += x[row, j]
            end
            @inbounds o[row, col] = acc
        end
    end
    return nothing
end

function kernel_segmented_sum_forw_w_aligned!(o, x, ψ, w, bags)
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
            @inbounds for j in bag
                acc += w[j] * x[row, j]
            end
            @inbounds o[row, col] = acc
        end
    end
    return nothing
end

function kernel_segmented_sum_back_aligned!(dx, Δ, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(Δ)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if !isempty(bag)
            @inbounds for j in bag
                CUDA.@atomic dx[row, j] += Δ[row, col]
            end
        end
    end
    return nothing
end

function kernel_segmented_sum_back_w_aligned!(dx, dw, Δ, x, w, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(Δ)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        bag = bags[col]
        if !isempty(bag)
            @inbounds for j in bag
                CUDA.@atomic dx[row, j] += w[j] * Δ[row, col]
                CUDA.@atomic dw[j] += Δ[row, col] * x[row, j]
            end
        end
    end
    return nothing
end

# CuAlignedBags forward dispatches
function segmented_sum_forw(x::CuMatrix, ψ::CuVector, bags::CuAlignedBags, w::Nothing)
    o = CUDA.zeros(eltype(x), size(x, 1), length(bags.bags))
    n_elements = length(o)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_sum_forw_aligned!(o, x, ψ, bags.bags)
    end
    o
end

function segmented_sum_forw(x::CuMatrix, ψ::CuVector, bags::CuAlignedBags, w::CuVector)
    o = CUDA.zeros(eltype(x), size(x, 1), length(bags.bags))
    n_elements = length(o)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_sum_forw_w_aligned!(o, x, ψ, w, bags.bags)
    end
    o
end

segmented_sum_forw(::Missing, ψ::CuVector, bags::CuAlignedBags, w) = repeat(ψ, 1, length(bags.bags))

# CuAlignedBags backward dispatches
function segmented_sum_back(Δ::CuMatrix, y, x::CuMatrix, ψ::CuVector, bags::CuAlignedBags, w::Nothing)
    dx = CUDA.zeros(eltype(x), size(x))
    dψ = CUDA.zeros(eltype(ψ), length(ψ))

    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_sum_back_aligned!(dx, Δ, bags.bags)
        @cuda threads=threads blocks=blocks kernel_sum_to_ψ!(dψ, Δ, bags.bags)
    end

    dx, dψ, ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent()
end

function segmented_sum_back(Δ::CuMatrix, y, x::CuMatrix, ψ::CuVector, bags::CuAlignedBags, w::CuVector)
    dx = CUDA.zeros(eltype(x), size(x))
    dw = CUDA.zeros(eltype(w), length(w))
    dψ = CUDA.zeros(eltype(ψ), length(ψ))

    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_segmented_sum_back_w_aligned!(dx, dw, Δ, x, w, bags.bags)
        @cuda threads=threads blocks=blocks kernel_sum_to_ψ!(dψ, Δ, bags.bags)
    end

    dx, dψ, ChainRulesCore.NoTangent(), dw
end

function segmented_sum_back(Δ::CuMatrix, y, x::Missing, ψ::CuVector, bags::CuAlignedBags, w)
    dψ = CUDA.zeros(eltype(ψ), length(ψ))
    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_sum_to_ψ!(dψ, Δ, bags.bags)
    end
    ChainRulesCore.ZeroTangent(), dψ, ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent()
end

# CuAlignedBags rrules
function ChainRulesCore.rrule(::typeof(segmented_sum_forw), x::CuMatrix, ψ::CuVector, bags::CuAlignedBags, w)
    y = segmented_sum_forw(x, ψ, bags, w)
    function segmented_sum_pullback(Δ)
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        grads = segmented_sum_back(Δ_unthunked, y, x, ψ, bags, w)
        (ChainRulesCore.NoTangent(), grads...)
    end
    y, segmented_sum_pullback
end

function ChainRulesCore.rrule(::typeof(segmented_sum_forw), x::Missing, ψ::CuVector, bags::CuAlignedBags, w)
    y = segmented_sum_forw(x, ψ, bags, w)
    function segmented_sum_pullback(Δ)
        Δ_unthunked = ChainRulesCore.unthunk(Δ)
        grads = segmented_sum_back(Δ_unthunked, y, x, ψ, bags, w)
        (ChainRulesCore.NoTangent(), grads...)
    end
    y, segmented_sum_pullback
end
