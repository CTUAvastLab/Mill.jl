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

function reduce(::typeof(catobs), as::Vector{T}) where {T<:BagNode}
    data = reduce(catobs, [x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    bags = vcat((d.bags for d in as)...)
    BagNode(data, bags, metadata)
end

function Base.getindex(x::BagNode, i::VecOrRange)
    nb, ii = remapbag(x.bags, i)
    isempty(ii) && return(BagNode(nothing, nb, nothing))
    BagNode(subset(x.data,ii), nb, subset(x.metadata, i))
end

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


# additional stuff to handle empty bags 
function reduce(::typeof(catobs), as::U) where {U<:Vector{BagNode{T,AlignedBags,Nothing} where T}}
    nonmissing_as = filter(x -> x.data != nothing, as)
    data = reduce(catobs, [x.data for x in nonmissing_as])
    nonmissing_meta = filter(x -> x.metadata != nothing, as)
    metadata = isempty(nonmissing_meta) ? nothing : reduce(catobs, [a.metadata for a in nonmissing_meta])
    bags = vcat((d.bags for d in as)...)
    BagNode(data, bags, metadata)
end
