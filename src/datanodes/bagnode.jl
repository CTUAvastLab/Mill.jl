"""
    BagNode{T <: Union{AbstractMillNode, Missing}, B <: AbstractBags, C} <: AbstractBagNode

Data node that represents a multi-instance learning problem.

Contains instances stored in a subtree of type `T`, bag indices of type `B` and optional metadata of type `C`.

See also: [`WeightedBagNode`](@ref), [`AbstractBagNode`](@ref),
    [`AbstractMillNode`](@ref), [`BagModel`](@ref).
"""
struct BagNode{T <: Maybe{AbstractMillNode}, B <: AbstractBags, C} <: AbstractBagNode
    data::T
    bags::B
    metadata::C

    function BagNode(d::T, b::B, m::C=nothing) where {T <: Maybe{AbstractMillNode}, B <: AbstractBags, C}
        @assert(!ismissing(d) || all(length.(b) .== 0),
                "BagNode with `missing` in data cannot have a non-empty bag")
        @assert(ismissing(d) || numobs(d) ≥ maxindex(b),
                "Bag indices range is greater than number of observations")
        new{T, B, C}(d, b, m)
    end
end

"""
    BagNode(d, b, m=nothing)

Construct a new [`BagNode`](@ref) with data `d`, bags `b`, and metadata `m`.

`d` is either an [`AbstractMillNode`](@ref) or `missing`. Any other type is wrapped in an [`ArrayNode`](@ref).

If `b` is an `AbstractVector`, [`Mill.bags`](@ref) is applied first.

# Examples
```jldoctest
julia> BagNode(ArrayNode(maybehotbatch([1, missing, 2], 1:2)), AlignedBags([1:1, 2:3]))
BagNode  2 obs
  ╰── ArrayNode(2×3 MaybeHotMatrix with Union{Missing, Bool} elements)  3 obs

julia> BagNode(randn(2, 5), [1, 2, 2, 1, 1])
BagNode  2 obs
  ╰── ArrayNode(2×5 Array with Float64 elements)  5 obs
```

See also: [`WeightedBagNode`](@ref), [`AbstractBagNode`](@ref),
    [`AbstractMillNode`](@ref), [`BagModel`](@ref).
"""
BagNode(d::Maybe{AbstractMillNode}, b::AbstractVector, m=nothing) = BagNode(d, bags(b), m)
BagNode(d, b, m=nothing) = BagNode(_arraynode(d), b, m)

Flux.@layer :ignore BagNode

mapdata(f, x::BagNode) = BagNode(mapdata(f, x.data), x.bags, x.metadata)

dropmeta(x::BagNode) = BagNode(dropmeta(x.data), x.bags)

function Base.getindex(x::BagNode, i::VecOrRange{<:Integer})
    nb, ii = remapbags(x.bags, i)
    emptyismissing() && isempty(ii) && return(BagNode(missing, nb, nothing))
    BagNode(x.data[ii], nb, metadata_getindex(x, i))
end

function Base.reduce(::typeof(catobs), as::Vector{<:BagNode})
    BagNode(
        reduce(catobs, data.(as)),
        reduce(vcat, [n.bags for n in as]),
        reduce(catobs, metadata.(as))
    )
end

function removeinstances(a::BagNode, mask)
    BagNode(a.data[mask], adjustbags(a.bags, mask), a.metadata)
end

Base.hash(n::BagNode, h::UInt) = hash((n.data, n.bags, n.metadata), h)
(n1::BagNode == n2::BagNode) = n1.data == n2.data && n1.bags == n2.bags && n1.metadata == n2.metadata
Base.isequal(n1::BagNode, n2::BagNode) = isequal(n1.data, n2.data) && isequal(n1.bags, n2.bags) && isequal(n1.metadata, n2.metadata)
