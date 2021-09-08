"""
    BagNode{T <: Union{AbstractMillNode, Missing}, B <: AbstractBags, C} <: AbstractBagNode

Data node that represents a multi-instance learning problem.
Contains instances stored in a subtree of type `T`,
bag indices of type `B` and optional metadata of type `C`.

See also: [`WeightedBagNode`](@ref), [`AbstractBagNode`](@ref),
    [`AbstractMillNode`](@ref), [`BagModel`](@ref).
"""
struct BagNode{T <: Maybe{AbstractMillNode}, B <: AbstractBags, C} <: AbstractBagNode
    data::T
    bags::B
    metadata::C

    function BagNode(d::T, b::B, m::C=nothing) where {T <: Maybe{AbstractMillNode}, B <: AbstractBags, C}
        ismissing(d) && any(length.(b) .> 0) && error("BagNode with `missing` in data cannot have a non-empty bag")
        new{T, B, C}(d, b, m)
    end
end

"""
    BagNode(d::Union{AbstractMillNode, Missing}, b::AbstractBags, m=nothing)
    BagNode(d::Union{AbstractMillNode, Missing}, b::AbstractVector, m=nothing)

Construct a new [`BagNode`](@ref) with data `d`, bags `b`, and metadata `m`.
If `b` is an `AbstractVector`, [`Mill.bags`](@ref) is applied first.

# Examples
```jldoctest
julia> BagNode(ArrayNode(maybehotbatch([1, missing, 2], 1:2)), AlignedBags([1:1, 2:3]))
BagNode 	# 2 obs, 104 bytes
  └── ArrayNode(2×3 MaybeHotMatrix with Union{Missing, Bool} elements) 	# 3 obs, 87 bytes

julia> BagNode(ArrayNode(randn(2, 5)), [1, 2, 2, 1, 1])
BagNode 	# 2 obs, 200 bytes
  └── ArrayNode(2×5 Array with Float64 elements) 	# 5 obs, 128 bytes
```

See also: [`WeightedBagNode`](@ref), [`AbstractBagNode`](@ref),
    [`AbstractMillNode`](@ref), [`BagModel`](@ref).
"""
BagNode(d::Maybe{AbstractMillNode}, b::AbstractVector, m=nothing) = BagNode(d, bags(b), m)

Flux.@functor BagNode

mapdata(f, x::BagNode) = BagNode(mapdata(f, x.data), x.bags, x.metadata)

dropmeta(x::BagNode) = BagNode(x.data, x.bags)

function Base.getindex(x::BagNode, i::VecOrRange{<:Int})
    nb, ii = remapbags(x.bags, i)
    emptyismissing() && isempty(ii) && return(BagNode(missing, nb, nothing))
    BagNode(subset(x.data,ii), nb, subset(x.metadata, i))
end

function reduce(::typeof(catobs), as::Vector{<:BagNode})
    d = filter(!ismissing, data.(as))
    md = filter(!isnothing, metadata.(as))
    bags = reduce(vcat, [n.bags for n in as])
    BagNode(reduce(catobs, d), bags, reduce(catobs, md))
end

removeinstances(a::BagNode, mask) = BagNode(subset(a.data, findall(mask)), adjustbags(a.bags, mask), a.metadata)

Base.hash(e::BagNode, h::UInt) = hash((e.data, e.bags, e.metadata), h)
(e1::BagNode == e2::BagNode) = e1.data == e2.data && e1.bags == e2.bags && e1.metadata == e2.metadata
Base.isequal(e1::BagNode, e2::BagNode) = isequal(e1.data, e2.data) && isequal(e1.bags, e2.bags) && isequal(e1.metadata, e2.metadata)
