struct WeightedBagNode{T <: Union{Missing, AbstractNode}, B <: AbstractBags, W, C} <: AbstractBagNode
    data::T
    bags::B
    weights::Vector{W}
    metadata::C

    function WeightedBagNode(d::T, b::B, w::Vector{W}, m::C=nothing) where {T <: Union{Missing, AbstractNode}, B <: AbstractBags, W, C}
        ismissing(d) && any(_len.(b.bags) .!= 0) && error("WeightedBagNode with nothing in data cannot have a non-empty bag")
        new{T, B, W, C}(d, b, w, m)
    end
end

WeightedBagNode(x::T, b::Vector, weights::Vector{W}, metadata::C=nothing) where {T, W, C} =
WeightedBagNode(x, bags(b), weights, metadata)

mapdata(f, x::WeightedBagNode) = WeightedBagNode(mapdata(f, x.data), x.bags, x.weights, x.metadata)

function Base.getindex(x::WeightedBagNode{T, B, W}, i::VecOrRange) where {T, B, W}
    nb, ii = remapbag(x.bags, i)
    _emptyismissing[] && isempty(ii) && return(WeightedBagNode(missing, nb, W[], nothing))
    WeightedBagNode(subset(x.data,ii), nb, subset(x.weights, ii), subset(x.metadata, ii))
end

function reduce(::typeof(catobs), as::Vector{T}) where {T <: WeightedBagNode}
    data = filter(!ismissing, [x.data for x in as])
    metadata = filter(!isnothing, [x.metadata for x in as])
    bags = vcat((d.bags for d in as)...)
    WeightedBagNode(isempty(data) ? missing : reduce(catobs, data),
                    bags, vcat((a.weights for a in as)...),
            isempty(metadata) ? nothing : reduce(catobs, metadata))
end

removeinstances(a::WeightedBagNode, mask) = WeightedBagNode(subset(a.data, findall(mask)), adjustbags(a.bags, mask), subset(a.weights, findall(mask)), a.metadata)

Base.hash(e::WeightedBagNode{T,B,W,C}, h::UInt) where {T,B,W,C} = hash((T,B,W,C, e.data, e.bags, e.weights, e.metadata), h)
Base.:(==)(e1::WeightedBagNode{T,B,W,C}, e2::WeightedBagNode{T,B,W,C}) where {T,B,W,C} = e1.data == e2.data && e1.bags == e2.bags && e1.weights == e2.weights && e1.metadata == e2.metadata
