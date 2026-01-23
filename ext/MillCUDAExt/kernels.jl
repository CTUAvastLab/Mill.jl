# Shared CUDA kernel utilities and helper functions

const CUDA_THREADS = 256

# Helper to get bag start/stop from CuArray of UnitRange
@inline bag_start(bags, bi) = bags[bi].start
@inline bag_stop(bags, bi) = bags[bi].stop
@inline bag_length(bags, bi) = bag_stop(bags, bi) - bag_start(bags, bi) + 1

# Check if a bag is empty (has length <= 0)
@inline isbagempty_kernel(bags, bi) = bag_stop(bags, bi) < bag_start(bags, bi)

"""
    compute_grid_config(n_elements)

Compute CUDA grid configuration for n_elements operations.
Returns (threads, blocks) tuple.
"""
function compute_grid_config(n_elements::Int)
    threads = min(CUDA_THREADS, n_elements)
    blocks = cld(n_elements, threads)
    return threads, blocks
end

"""
    kernel_fill_missing!(o, ψ, bags)

CUDA kernel to fill output matrix with missing value ψ for empty bags.
Each thread handles one (row, bag) element.
"""
function kernel_fill_missing!(o, ψ, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(o)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        if isbagempty_kernel(bags, col)
            @inbounds o[row, col] = ψ[row]
        end
    end
    return nothing
end

"""
    kernel_sum_to_ψ!(dψ, Δ, bags)

CUDA kernel for backward pass: accumulate gradients into dψ for empty bags.
Uses atomic operations for thread-safe accumulation.
"""
function kernel_sum_to_ψ!(dψ, Δ, bags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_rows, n_cols = size(Δ)

    if idx <= n_rows * n_cols
        row = ((idx - 1) % n_rows) + 1
        col = ((idx - 1) ÷ n_rows) + 1

        if isbagempty_kernel(bags, col)
            @inbounds CUDA.@atomic dψ[row] += Δ[row, col]
        end
    end
    return nothing
end

"""
    missingbags_mapreducedim!(f, op, dψ, Δ, bags)

Accumulate gradients from empty bags into dψ using map-reduce.
"""
function missingbags_mapreducedim!(f, op, dψ, Δ, bags)
    n_elements = length(Δ)
    if n_elements > 0
        threads, blocks = compute_grid_config(n_elements)
        @cuda threads=threads blocks=blocks kernel_sum_to_ψ!(dψ, Δ, bags)
    end
    return dψ
end
