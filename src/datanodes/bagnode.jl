abstract type AbstractBagNode{T <: AbstractNode, C} <: AbstractNode end


mutable struct BagNode{T, B <: AbstractBags, C} <: AbstractBagNode{T, C}
    data::T
    bags::B
    metadata::C

    function BagNode(data::T, b::B, metadata::C) where {T <: AbstractNode, B <: AbstractBags, C}
        new{T, B, C}(data, b, metadata)
    end
end

BagNode(data::T, b::Union{AbstractBags, Vector}, metadata::C=nothing) where {T <: AbstractNode, C} =
BagNode(data, bags(b), metadata)

mapdata(f, x::BagNode) = BagNode(mapdata(f, x.data), x.bags, x.metadata)

Base.ndims(x::BagNode) = 0
LearnBase.nobs(a::AbstractBagNode) = length(a.bags)
LearnBase.nobs(a::AbstractBagNode, ::Type{ObsDim.Last}) = nobs(a)

function reduce(::typeof(catobs), as::Vector{T}) where {T<:BagNode}
    data = reduce(catobs, [x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    bags = vcat((d.bags for d in as)...)
    BagNode(data, bags, metadata)
end

function Base.getindex(x::BagNode, i::VecOrRange)
    nb, ii = remapbag(x.bags, i)
    BagNode(subset(x.data,ii), nb, subset(x.metadata, i))
end

removeinstances(a::BagNode, mask) = BagNode(subset(a.data, findall(mask)), adjustbags(a.bags, mask), a.metadata)

adjustbags(bags::AlignedBags, mask::T) where {T<:Union{Vector{Bool}, BitArray{1}}} = length2bags(map(b -> sum(@view mask[b]), bags))

function dsprint(io::IO, n::BagNode{ArrayNode}; pad=[], s="", tr=false)
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io,"BagNode$(size(n.data)) with $(length(n.bags)) bag(s)$(tr_repr(s, tr))\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.data, pad = [pad; (c, "      ")], s=s * encode(1, 1), tr=tr)
end

function dsprint(io::IO, n::BagNode; pad=[], s="", tr=false)
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io,"BagNode with $(length(n.bags)) bag(s)$(tr_repr(s, tr))\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.data, pad = [pad; (c, "      ")], s=s * encode(1, 1), tr=tr)
end
