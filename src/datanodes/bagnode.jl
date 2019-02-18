mutable struct BagNode{T, B <: AbstractBags, C} <: AbstractBagNode
    data::T
    bags::B
    metadata::C
end

BagNode(data, b::Vector, metadata = nothing) = BagNode(data, bags(b), metadata)

mapdata(f, x::BagNode) = BagNode(mapdata(f, x.data), x.bags, x.metadata)

Base.ndims(x::BagNode) = 0
LearnBase.nobs(a::AbstractBagNode) = length(a.bags)
LearnBase.nobs(a::AbstractBagNode, ::Type{ObsDim.Last}) = nobs(a)

function Base.getindex(x::BagNode, i::VecOrRange)
    nb, ii = remapbag(x.bags, i)
    isempty(ii) && return(BagNode(nothing, nb, nothing))
    BagNode(subset(x.data,ii), nb, subset(x.metadata, i))
end


# additional stuff to handle empty bags 
"""
    verifymissing(a)

    verify that if a is nothing than only empty bags are present

"""
function verifymissing(a)
    a.data != nothing && return(true)
    any(len.(a.bags.bags) .!= 0) && @error "a missing data has non-empty bag "
    return(true)
end

len(a::UnitRange) = max(a.stop - a.start + 1, 0)

function reducenonmissing(a, key) 
    o = filter(!isnothing, [getproperty(x, key) for x in a])
    isempty(o) ? nothing : reduce(catobs, o)
end

function reduce(::typeof(catobs), as::Vector{T}) where {T<:BagNode}
    all(verifymissing.(as))
    data = reducenonmissing(as, :data)
    metadata = reducenonmissing(as, :metadata)
    bags = vcat((d.bags for d in as)...)
    BagNode(data, bags, metadata)
end

Base.cat(a::BagNode, b::BagNode; dims = Colon) = reduce(catobs, [a, b])

removeinstances(a::BagNode, mask) = BagNode(subset(a.data, findall(mask)), adjustbags(a.bags, mask), a.metadata)

adjustbags(bags::AlignedBags, mask::T) where {T<:Union{Vector{Bool}, BitArray{1}}} = length2bags(map(b -> sum(@view mask[b]), bags))

function dsprint(io::IO, n::BagNode{T, B, C}; pad=[]) where {T <:AbstractNode, B, C}
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io,"BagNode with $(length(n.bags)) bag(s)\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.data, pad = [pad; (c, "      ")])
end

function dsprint(io::IO, n::BagNode{T, B, C}; pad=[]) where {T <:Nothing, B, C}
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io,"BagNode with $(length(n.bags)) empty bag(s)\n", color=c)
end