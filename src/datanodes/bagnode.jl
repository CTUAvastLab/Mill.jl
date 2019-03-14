mutable struct BagNode{T <: Union{Nothing, Mill.AbstractNode}, B <: AbstractBags, C} <: AbstractBagNode
    data::T
    bags::B
    metadata::C
    function BagNode(d::T, b::B, m::C) where {T <: Union{Nothing, Mill.AbstractNode}, B <: AbstractBags, M}
        isnothing(d) && any(_len.(b.bags) .!= 0) && error("BagNode with nothing in data cannot have a non-empty bag")
        new{T, B, C}(d, b, m)
    end
end

_len(a::UnitRange) = max(a.stop - a.start + 1, 0)
_len(a::Vector) = length(a)

BagNode(data::T, b::Vector, metadata::M = nothing) where {T, M} = BagNode(data, bags(b), metadata)

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
function reducenonmissing(a, key) 
    o = filter(!isnothing, [getproperty(x, key) for x in a])
    isempty(o) ? nothing : reduce(catobs, o)
end

function reduce(::typeof(catobs), as::Vector{T}) where {T<:BagNode}
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
