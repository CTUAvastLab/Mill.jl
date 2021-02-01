struct BagNode{T <: Union{Missing, AbstractNode}, B <: AbstractBags, C} <: AbstractBagNode
    data::T
    bags::B
    metadata::C

    function BagNode(d::T, b::B, m::C=nothing) where {T <: Maybe{AbstractNode}, B <: AbstractBags, C}
        ismissing(d) && any(length.(b.bags) .> 0) && error("BagNode with nothing in data cannot have a non-empty bag")
        new{T, B, C}(d, b, m)
    end
end

BagNode(data::T, b::Vector, metadata::C=nothing) where {T, C} = BagNode(data, bags(b), metadata)

Flux.@functor BagNode

mapdata(f, x::BagNode) = BagNode(mapdata(f, x.data), x.bags, x.metadata)

function Base.getindex(x::BagNode, i::VecOrRange{<:Int})
    nb, ii = remapbags(x.bags, i)
    emptyismissing() && isempty(ii) && return(BagNode(missing, nb, nothing))
    BagNode(subset(x.data,ii), nb, subset(x.metadata, i))
end

function reduce(::typeof(catobs), as::Vector{T}) where {T <: BagNode}
    d = filter(!ismissing, data.(as))
    md = filter(!isnothing, metadata.(as))
    bags = _catbags([n.bags for n in as])
    BagNode(isempty(d) ? missing : reduce(catobs, d),
            bags,
            isempty(md) ? nothing : reduce(catobs, md))
end

removeinstances(a::BagNode, mask) = BagNode(subset(a.data, findall(mask)), adjustbags(a.bags, mask), a.metadata)

Base.hash(e::BagNode, h::UInt) = hash((e.data, e.bags, e.metadata), h)
(e1::BagNode == e2::BagNode) = e1.data == e2.data && e1.bags == e2.bags && e1.metadata == e2.metadata
Base.isequal(e1::BagNode, e2::BagNode) = isequal(e1.data, e2.data) && isequal(e1.bags, e2.bags) && isequal(e1.metadata, e2.metadata)
