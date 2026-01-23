# CompressedBags - GPU-friendly representation of bag indices
# Flattens variable-length bags into contiguous arrays for efficient GPU access

"""
    CompressedBags{T, I<:AbstractVector{T}, B<:AbstractVector{UnitRange{T}}}

GPU-friendly representation of bag indices. Stores:
- `indices`: flattened vector of all instance indices
- `bags`: vector of ranges into `indices` for each bag
- `num_observations`: total number of observations (instances)

This structure allows efficient parallel access on GPU where each thread block
can process one bag by reading from the corresponding range in `indices`.
"""
struct CompressedBags{T, I<:AbstractVector{T}, B<:AbstractVector{UnitRange{T}}} <: AbstractBags{T}
    indices::I
    bags::B
    num_observations::T
end

const CuCompressedBags{T} = CompressedBags{T, <:CuArray, <:CuArray}

# Forward common operations to the bags field
Flux.@forward CompressedBags.bags Base.firstindex, Base.lastindex, Base.eachindex, Base.length

Base.getindex(b::CompressedBags, i::Int) = @view b.indices[b.bags[i]]
Base.getindex(b::CompressedBags, I::AbstractUnitRange{<:Integer}) = [b[i] for i in I]

Base.first(b::CompressedBags) = b[1]
function Base.first(b::CompressedBags, n::Int)
    n < 0 && throw(ArgumentError("Number of elements must be nonnegative."))
    n > length(b) && return b[1:end]
    b[1:n]
end

Base.last(b::CompressedBags) = b[end]
function Base.last(b::CompressedBags, n::Int)
    n < 0 && throw(ArgumentError("Number of elements must be nonnegative."))
    n > length(b) && return b[1:end]
    b[end-n+1:end]
end

function Base.iterate(b::CompressedBags, i=1)
    i > length(b) && return nothing
    b[i], i + 1
end

Base.isempty(b::CompressedBags) = isempty(b.bags)

Mill.numobs(b::CompressedBags) = length(b.bags)
maxindex(b::CompressedBags) = isempty(b) ? -1 : b.num_observations

"""
    CompressedBags(bags::Union{AlignedBags, ScatteredBags})

Convert CPU bag representation to CompressedBags format suitable for GPU transfer.
"""
function CompressedBags(bags::AlignedBags{T}) where T
    indices = reduce(vcat, [collect(b) for b in bags.bags]; init=T[])
    if isempty(indices)
        return CompressedBags(indices, UnitRange{T}[], zero(T))
    end
    bs = Mill.length2bags(length.(bags.bags))
    CompressedBags(indices, bs.bags, T(maximum(indices)))
end

function CompressedBags(bags::ScatteredBags{T}) where T
    indices = reduce(vcat, [collect(b) for b in bags.bags]; init=T[])
    if isempty(indices)
        return CompressedBags(indices, UnitRange{T}[], zero(T))
    end
    bs = Mill.length2bags(length.(bags.bags))
    CompressedBags(indices, bs.bags, T(maximum(indices)))
end

"""
    Flux.gpu(b::CompressedBags)

Move CompressedBags to GPU, converting to Int32 for GPU efficiency.
"""
function Flux.gpu(b::CompressedBags)
    CompressedBags(
        CuArray(Int32.(b.indices)),
        CuArray([Int32(r.start):Int32(r.stop) for r in b.bags]),
        Int32(b.num_observations)
    )
end

"""
    Flux.gpu(b::Union{AlignedBags, ScatteredBags})

Move CPU bags to GPU by first converting to CompressedBags format.
"""
Flux.gpu(b::AlignedBags) = gpu(CompressedBags(b))
Flux.gpu(b::ScatteredBags) = gpu(CompressedBags(b))

"""
    Flux.cpu(b::CuCompressedBags)

Move CompressedBags back to CPU.
"""
function Flux.cpu(b::CuCompressedBags)
    CompressedBags(
        Array(b.indices),
        Array(b.bags),
        b.num_observations
    )
end

# Helper function to check if a bag is empty
@inline isbagempty(bags::AbstractVector{<:UnitRange}, bi::Int) = isempty(bags[bi])
