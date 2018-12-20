
mutable struct WeightedBagNode{T, W, C} <: AbstractBagNode{T, C}
    data::T
    bags::Bags
    weights::Vector{W}
    metadata::C

    function WeightedBagNode{T, W, C}(
                                      data::T, bags::Union{Bags, Vector}, weights::Vector{W}, metadata::C) where {T <: AbstractNode, W, C}
        new(data, bag(bags), weights, metadata)
    end
end

WeightedBagNode(x::T, b::Union{Bags, Vector}, weights::Vector{W}, metadata::C=nothing) where {T <: AbstractNode, W, C} =
WeightedBagNode{T, W, C}(x, b, weights, metadata)

mapdata(f, x::WeightedBagNode) = WeightedBagNode(mapdata(f, x.data), x.bags, x.weights, x.metadata)

Base.ndims(x::WeightedBagNode) = 0

function reduce(::typeof(catobs), as::Vector{T}) where {T<:WeightedBagNode}
    data = reduce(catobs, [x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    bags = _catbags([d.bags for d in as])
    weights = reduce(catobs, [d.weights for d in as])
    WeightedBagNode(data, bags, weights, metadata)
end

function Base.getindex(x::WeightedBagNode, i::VecOrRange)
    nb, ii = remapbag(x.bags, i)
    WeightedBagNode(subset(x.data,ii), nb, subset(x.weights, ii), subset(x.metadata, ii))
end

function dsprint(io::IO, n::WeightedBagNode{ArrayNode}; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "WeightedNode$(size(n.data)) with $(length(n.bags)) bag(s) and weights Σw = $(sum(n.weights))\n", color=c)
end

function dsprint(io::IO, n::WeightedBagNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "WeightedNode with $(length(n.bags)) bag(s) and weights Σw = $(sum(n.weights))\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.data, pad = [pad; (c, "      ")])
end
