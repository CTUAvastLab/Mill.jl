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
    nb, ii = remapbag(x.bags, i)
    _emptyismissing[] && isempty(ii) && return(BagNode(missing, nb, nothing))
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

Base.hash(e::BagNode{T,B,C}, h::UInt) where {T,B,C} = hash((T, B, C, e.data, e.bags, e.metadata), h)
(e1::BagNode{T,B,C} == e2::BagNode{T,B,C}) where {T,B,C} =
    e1.data == e2.data && e1.bags == e2.bags && e1.metadata == e2.metadata
