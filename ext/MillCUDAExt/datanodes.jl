# GPU data movement for Mill data nodes

using Mill: ArrayNode, BagNode, WeightedBagNode, ProductNode, LazyNode
using Mill: MaybeHotVector, MaybeHotMatrix, NGramMatrix
using Mill: PreImputingMatrix, PostImputingMatrix
import Mill: _mul
# Type alias for GPU MaybeHotMatrix (defined early since used in type signatures below)
const CuMaybeHotMatrix{T, U} = MaybeHotMatrix{T, <:CUDA.CUDA.CuVector{T}, U}

# ArrayNode GPU movement
function Flux.gpu(ds::ArrayNode)
    ArrayNode(gpu(ds.data), ds.metadata)
end

function Flux.cpu(ds::ArrayNode)
    ArrayNode(cpu(ds.data), ds.metadata)
end


# BagNode GPU movement
function Flux.gpu(ds::BagNode)
    BagNode(gpu(ds.data), gpu(ds.bags), ds.metadata)
end

function Flux.cpu(ds::BagNode)
    BagNode(cpu(ds.data), cpu(ds.bags), ds.metadata)
end

# WeightedBagNode GPU movement
function Flux.gpu(ds::WeightedBagNode)
    WeightedBagNode(gpu(ds.data), gpu(ds.bags), gpu(ds.weights), ds.metadata)
end

function Flux.cpu(ds::WeightedBagNode)
    WeightedBagNode(cpu(ds.data), cpu(ds.bags), cpu(ds.weights), ds.metadata)
end

# ProductNode GPU movement
function Flux.gpu(ds::ProductNode)
    ProductNode(map(gpu, ds.data), ds.metadata)
end

function Flux.cpu(ds::ProductNode)
    ProductNode(map(cpu, ds.data), ds.metadata)
end

# Special arrays GPU support

# MaybeHotVector - single index, keep on CPU (negligible data)
Adapt.adapt_structure(to, x::MaybeHotVector) = x

# MaybeHotMatrix - move indices to GPU
# function Adapt.adapt_structure(to, x::MaybeHotMatrix)
#     MaybeHotMatrix(Adapt.adapt_structure(to, x.I), x.l)
# end

function Flux.cpu(x::MaybeHotMatrix)
    MaybeHotMatrix(Flux.cpu(x.I), x.l)
end

function Flux.gpu(x::MaybeHotMatrix)
    MaybeHotMatrix(cu(x.I), x.l)
end

# Multiplication: CUDA.CuMatrix * MaybeHotVector (column selection) - indices on CPU
function Base.:*(A::CUDA.CuMatrix{T}, b::MaybeHotVector{<:Integer}) where T
    A[:, b.i]
end

function Base.:*(A::CUDA.CuMatrix{T}, b::MaybeHotVector{Missing}) where T
    CUDA.fill(missing, size(A, 1))
end

# Multiplication: CUDA.CuMatrix * CuMaybeHotMatrix (indices on GPU)
function Base.:*(A::CUDA.CuMatrix{T}, B::CuMaybeHotMatrix{<:Integer, <:Any}) where T
    # Pure integer indices - efficient GPU gather
    A[:, B.I]
end

function Base.:*(A::CUDA.CuMatrix{T}, B::CuMaybeHotMatrix{Missing, <:Any}) where T
    CUDA.fill(missing, size(A, 1), length(B.I))
end

function Base.:*(A::CUDA.CuMatrix{T}, B::CuMaybeHotMatrix{<:Mill.Maybe{<:Integer}, <:Any}) where T
    _mul(A, B)
end

# GPU kernel for mixed missing/non-missing multiplication
function _mul(A::CuArray{T, 2}, B::MaybeHotMatrix{U, V} where {U<:Union{Missing, UInt32}, V<:(CuArray{U, 1})}) where {T}
    m, n = size(A, 1), length(B.I)
    result = CUDA.zeros(T, m, n)

    # Use GPU broadcast to handle the gathering
    # For each column j: if B.I[j] is not missing, copy A[:, B.I[j]], else leave as zero
    kernel = CUDA.@cuda launch=false _gather_kernel!(result, A, B.I)
    config = CUDA.launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    kernel(result, A, B.I; threads, blocks)

    result
end

function _gather_kernel!(result, A, indices)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= length(indices)
        idx = indices[j]
        if !ismissing(idx)
            for i in 1:size(A, 1)
                result[i, j] = A[i, idx]
            end
        end
    end
    return nothing
end

# Gradient rule for CUDA.CuMatrix * CuMaybeHotMatrix{Maybe}
function ChainRulesCore.rrule(::typeof(*), A::CUDA.CuMatrix{T}, B::CuMaybeHotMatrix{<:Any, <:Any}) where T
    result = _mul(A, B)

    function pullback(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        dA = CUDA.zeros(T, size(A))

        # Scatter-add gradient back to dA
        kernel = CUDA.@cuda launch=false _scatter_add_kernel!(dA, Δ, B.I)
        config = CUDA.launch_configuration(kernel.fun)
        threads = min(length(B.I), config.threads)
        blocks = cld(length(B.I), threads)
        kernel(dA, Δ, B.I; threads, blocks)

        return NoTangent(), dA, NoTangent()
    end

    return result, pullback
end

function _scatter_add_kernel!(dA, Δ, indices)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= length(indices)
        idx = indices[j]
        if !ismissing(idx)
            for i in 1:size(dA, 1)
                CUDA.@atomic dA[i, idx] += Δ[i, j]
            end
        end
    end
    return nothing
end

# Also support CPU MaybeHotMatrix * CUDA.CuMatrix for convenience (will be slower)
function Base.:*(A::CUDA.CuMatrix{T}, B::MaybeHotMatrix{<:Integer}) where T
    A[:, B.I]
end

function Base.:*(A::CUDA.CuMatrix{T}, B::MaybeHotMatrix{Missing}) where T
    CUDA.fill(T(NaN), size(A, 1), length(B.I))
end

function Base.:*(A::CUDA.CuMatrix{T}, B::MaybeHotMatrix{<:Mill.Maybe{<:Integer}}) where T
    # Convert to GPU and multiply
    A * gpu(B)
end

# NGramMatrix - convert to dense for GPU operations
# NGramMatrix is inherently sparse, so we convert to dense CUDA.CuMatrix
function Adapt.adapt_structure(to, x::NGramMatrix)
    # Convert NGramMatrix to dense matrix then move to GPU
    dense = Matrix(x)
    CUDA.CuArray(dense)
end

# ImputingMatrix GPU support
function Adapt.adapt_structure(to, x::PreImputingMatrix)
    PreImputingMatrix(Adapt.adapt_structure(to, x.W), Adapt.adapt_structure(to, x.ψ))
end

function Adapt.adapt_structure(to, x::PostImputingMatrix)
    PostImputingMatrix(Adapt.adapt_structure(to, x.W), Adapt.adapt_structure(to, x.ψ))
end

# PostImputingMatrix * MaybeHotVector on GPU (MaybeHotVector stays on CPU - single index)
function Base.:*(A::PostImputingMatrix{T, <:CUDA.CuMatrix, <:CUDA.CuVector}, b::MaybeHotVector{<:Integer}) where T
    A.W[:, b.i]
end

function Base.:*(A::PostImputingMatrix{T, <:CUDA.CuMatrix, <:CUDA.CuVector}, b::MaybeHotVector{Missing}) where T
    A.ψ
end

# PostImputingMatrix * CuMaybeHotMatrix on GPU
function Base.:*(A::PostImputingMatrix{T, <:CUDA.CuMatrix, <:CUDA.CuVector}, B::CuMaybeHotMatrix{<:Integer, <:Any}) where T
    A.W[:, B.I]
end

function Base.:*(A::PostImputingMatrix{T, <:CUDA.CuMatrix, <:CUDA.CuVector}, B::CuMaybeHotMatrix{Missing, <:Any}) where T
    repeat(A.ψ, 1, length(B.I))
end

function Base.:*(A::PostImputingMatrix{T, <:CUDA.CuMatrix, <:CUDA.CuVector}, B::CuMaybeHotMatrix{<:Mill.Maybe{<:Integer}, <:Any}) where T
    _postimpute_cumaybehot_gpu(A, B)
end

# GPU kernel-based implementation for PostImputingMatrix * CuMaybeHotMatrix{Maybe}
function _postimpute_cumaybehot_gpu(A::PostImputingMatrix{T, <:CUDA.CuMatrix, <:CUDA.CuVector}, B::CuMaybeHotMatrix{<:Any, <:Any}) where T
    m, n = size(A.W, 1), length(B.I)
    result = CUDA.zeros(T, m, n)

    kernel = CUDA.@cuda launch=false _postimpute_gather_kernel!(result, A.W, A.ψ, B.I)
    config = CUDA.launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    kernel(result, A.W, A.ψ, B.I; threads, blocks)

    result
end

function _postimpute_gather_kernel!(result, W, ψ, indices)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= length(indices)
        idx = indices[j]
        if ismissing(idx)
            # Fill with ψ
            for i in 1:size(result, 1)
                result[i, j] = ψ[i]
            end
        else
            # Gather from W
            for i in 1:size(W, 1)
                result[i, j] = W[i, idx]
            end
        end
    end
    return nothing
end

# Gradient rule for PostImputingMatrix * CuMaybeHotMatrix{Maybe}
function ChainRulesCore.rrule(::typeof(*), A::PostImputingMatrix{T, <:CUDA.CuMatrix, <:CUDA.CuVector}, B::CuMaybeHotMatrix{<:Any, <:Any}) where T
    result = _postimpute_cumaybehot_gpu(A, B)

    function pullback(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        dW = CUDA.zeros(T, size(A.W))
        dψ = CUDA.zeros(T, size(A.ψ))

        # Scatter gradients back
        kernel = CUDA.@cuda launch=false _postimpute_scatter_kernel!(dW, dψ, Δ, B.I)
        config = CUDA.launch_configuration(kernel.fun)
        threads = min(length(B.I), config.threads)
        blocks = cld(length(B.I), threads)
        kernel(dW, dψ, Δ, B.I; threads, blocks)

        dA = Tangent{typeof(A)}(W = dW, ψ = dψ)
        return NoTangent(), dA, NoTangent()
    end

    return result, pullback
end

function _postimpute_scatter_kernel!(dW, dψ, Δ, indices)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= length(indices)
        idx = indices[j]
        if ismissing(idx)
            for i in 1:size(dψ, 1)
                CUDA.@atomic dψ[i] += Δ[i, j]
            end
        else
            for i in 1:size(dW, 1)
                CUDA.@atomic dW[i, idx] += Δ[i, j]
            end
        end
    end
    return nothing
end

# Also support CPU MaybeHotMatrix with GPU PostImputingMatrix (converts to GPU)
function Base.:*(A::PostImputingMatrix{T, <:CUDA.CuMatrix, <:CUDA.CuVector}, B::MaybeHotMatrix{<:Integer}) where T
    A.W[:, B.I]
end

function Base.:*(A::PostImputingMatrix{T, <:CUDA.CuMatrix, <:CUDA.CuVector}, B::MaybeHotMatrix{Missing}) where T
    repeat(A.ψ, 1, length(B.I))
end

function Base.:*(A::PostImputingMatrix{T, <:CUDA.CuMatrix, <:CUDA.CuVector}, B::MaybeHotMatrix{<:Mill.Maybe{<:Integer}}) where T
    A * gpu(B)
end

# # Aggregation operators GPU movement
# function Adapt.adapt_structure(to, a::Mill.SegmentedSum)
#     Mill.SegmentedSum(Adapt.adapt_structure(to, a.ψ))
# end

# function Adapt.adapt_structure(to, a::Mill.SegmentedMean)
#     Mill.SegmentedMean(Adapt.adapt_structure(to, a.ψ))
# end

# function Adapt.adapt_structure(to, a::Mill.SegmentedMax)
#     Mill.SegmentedMax(Adapt.adapt_structure(to, a.ψ))
# end

# function Adapt.adapt_structure(to, a::Mill.SegmentedPNorm)
#     Mill.SegmentedPNorm(Adapt.adapt_structure(to, a.ψ), Adapt.adapt_structure(to, a.ρ), Adapt.adapt_structure(to, a.c))
# end

# function Adapt.adapt_structure(to, a::Mill.SegmentedLSE)
#     Mill.SegmentedLSE(Adapt.adapt_structure(to, a.ψ), Adapt.adapt_structure(to, a.ρ))
# end

# function Adapt.adapt_structure(to, a::Mill.AggregationStack)
#     Mill.AggregationStack(map(gpu, a.fs))
# end

# function Adapt.adapt_structure(to, a::Mill.BagCount)
#     a  # BagCount has no parameters
# end
