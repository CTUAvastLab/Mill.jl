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
    isempty(ii) && return(WeightedBagNode(missing, nb, W[], nothing))
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

function dsprint(io::IO, n::WeightedBagNode{T}; pad=[], s="", tr=false) where T
    c = COLORS[(length(pad)%length(COLORS))+1]
    m = T <: Missing ? " missing " : " "
    paddedprint(io, "WeightedNode with $(length(n.bags))$(m)bag(s) and weights Σw = $(sum(n.weights))$(tr_repr(s, tr))\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.data, pad = [pad; (c, "      ")], s=s * encode(1, 1), tr=tr)
end
