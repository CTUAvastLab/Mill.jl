abstract type AbstractBagNode{T <: AbstractNode, C} <: AbstractNode end


mutable struct BagNode{T, C} <: AbstractBagNode{T, C}
    data::T
    bags::Bags
    metadata::C

    function BagNode{T, C}(data::T, bags::Union{Bags, Vector}, metadata::C) where {T <: AbstractNode, C}
        new(data, bag(bags), metadata)
    end
end

BagNode(x::T, b::Union{Bags, Vector}, metadata::C=nothing) where {T <: AbstractNode, C} =
BagNode{T, C}(x, b, metadata)


mapdata(f, x::BagNode) = BagNode(mapdata(f, x.data), x.bags, x.metadata)


Base.ndims(x::BagNode) = 0
LearnBase.nobs(a::AbstractBagNode) = length(a.bags)
LearnBase.nobs(a::AbstractBagNode, ::Type{ObsDim.Last}) = nobs(a)


function reduce(::typeof(catobs), as::Vector{T}) where {T<:BagNode}
    data = reduce(catobs, [x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    bags = _catbags([d.bags for d in as])
    BagNode(data, bags, metadata)
end


function Base.getindex(x::BagNode, i::VecOrRange)
    nb, ii = remapbag(x.bags, i)
    BagNode(subset(x.data,ii), nb, subset(x.metadata, i))
end


removeinstances(a::BagNode, mask) = BagNode(subset(a.data, findall(mask)), adjustbags(a.bags, mask), a.metadata)
adjustbags(bags::Vector{UnitRange{Int64}}, mask::T) where {T<:Union{Vector{Bool}, BitArray{1}}} = Mill.length2bags(map(b -> sum(@view mask[b]), bags))


function dsprint(io::IO, n::BagNode{ArrayNode}; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io,"BagNode$(size(n.data)) with $(length(n.bags)) bag(s)\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.data, pad = [pad; (c, "      ")])
end

function dsprint(io::IO, n::BagNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io,"BagNode with $(length(n.bags)) bag(s)\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.data, pad = [pad; (c, "      ")])
end
