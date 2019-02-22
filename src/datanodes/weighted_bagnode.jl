
mutable struct WeightedBagNode{T, B <: AbstractBags, W, C} <: AbstractBagNode{T, C}
    data::T
    bags::B
    weights::Vector{W}
    metadata::C

    function WeightedBagNode(data::T, b::B, weights::Vector{W}, metadata::C) where {T <: AbstractNode, B <: AbstractBags, W, C}
        new{T, B, W, C}(data, b, weights, metadata)
    end
end

WeightedBagNode(x::T, b::Union{AbstractBags, Vector}, weights::Vector{W}, metadata::C=nothing) where {T <: AbstractNode, W, C} =
WeightedBagNode(x, bags(b), weights, metadata)

mapdata(f, x::WeightedBagNode) = WeightedBagNode(mapdata(f, x.data), x.bags, x.weights, x.metadata)

Base.ndims(x::WeightedBagNode) = 0

function reduce(::typeof(catobs), as::Vector{T}) where {T<:WeightedBagNode}
    data = reduce(catobs, [x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    bags = vcat((d.bags for d in as)...)
    weights = reduce(catobs, [d.weights for d in as])
    WeightedBagNode(data, bags, weights, metadata)
end

function Base.getindex(x::WeightedBagNode, i::VecOrRange)
    nb, ii = remapbag(x.bags, i)
    WeightedBagNode(subset(x.data,ii), nb, subset(x.weights, ii), subset(x.metadata, ii))
end

function dsprint(io::IO, n::WeightedBagNode{ArrayNode}, pad=[], s="", tr=false)
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "WeightedNode$(size(n.data)) with $(length(n.bags)) bag(s) and weights Σw = $(sum(n.weights))$(repr(s, tr))\n", color=c)
end

function dsprint(io::IO, n::WeightedBagNode; pad=[], s="", tr=false)
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "WeightedNode with $(length(n.bags)) bag(s) and weights Σw = $(sum(n.weights))$(repr(s, tr))\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.data, pad = [pad; (c, "      ")], s=encode(s, 1, 1), tr=tr)
end
