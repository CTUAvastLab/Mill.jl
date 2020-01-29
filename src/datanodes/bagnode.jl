struct BagNode{T <: Union{Missing, AbstractNode}, B <: AbstractBags, C} <: AbstractBagNode
    data::T
    bags::B
    metadata::C

    function BagNode(d::T, b::B, m::C=nothing) where {T <: Union{Missing, AbstractNode}, B <: AbstractBags, C}
        ismissing(d) && any(_len.(b.bags) .!= 0) && error("BagNode with nothing in data cannot have a non-empty bag")
        new{T, B, C}(d, b, m)
    end
end

BagNode(data::T, b::Vector, metadata::C=nothing) where {T, C} = BagNode(data, bags(b), metadata)

Flux.@functor BagNode

mapdata(f, x::BagNode) = BagNode(mapdata(f, x.data), x.bags, x.metadata)

function Base.getindex(x::BagNode, i::VecOrRange)
    nb, ii = remapbag(x.bags, i)
    isempty(ii) && return(BagNode(missing, nb, nothing))
    BagNode(subset(x.data,ii), nb, subset(x.metadata, i))
end

function reduce(::typeof(catobs), as::Vector{T}) where {T <: BagNode}
    data = filter(!ismissing, [x.data for x in as])
    metadata = filter(!isnothing, [x.metadata for x in as])
    bags = _catbags([d.bags for d in as])
    BagNode(isempty(data) ? missing : reduce(catobs, data),
            bags,
            isempty(metadata) ? nothing : reduce(catobs, metadata))
end

removeinstances(a::BagNode, mask) = BagNode(subset(a.data, findall(mask)), adjustbags(a.bags, mask), a.metadata)

function dsprint(io::IO, n::BagNode{T}; pad=[], s="", tr=false) where T
    c = COLORS[(length(pad)%length(COLORS))+1]
    m = T <: Missing ? " missing " : " "
    paddedprint(io,"BagNode with $(length(n.bags))$(m)bag(s)$(tr_repr(s, tr))\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.data, pad = [pad; (c, "      ")], s=s * encode(1, 1), tr=tr)
end
