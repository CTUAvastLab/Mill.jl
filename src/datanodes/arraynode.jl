"""


"""
struct ArrayNode{A<:AbstractArray,C} <: AbstractNode
    data::A
    metadata::C
end

ArrayNode(data::AbstractArray) = ArrayNode(data, nothing)
# ArrayNode(data::AbstractNode, a...) = data

Flux.@functor ArrayNode

mapdata(f, x::ArrayNode) = ArrayNode(mapdata(f, x.data), x.metadata)

Base.ndims(x::ArrayNode) = Colon()
StatsBase.nobs(a::ArrayNode) = size(a.data, 2)
StatsBase.nobs(a::ArrayNode, ::Type{ObsDim.Last}) = nobs(a)

function reduce(::typeof(catobs), as::Vector{<:ArrayNode})
    data = reduce(catobs, [x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    ArrayNode(data, metadata)
end

function reduce(::typeof(vcat), as::Vector{<:ArrayNode})
    data = reduce(vcat, [a.data for a in as])
    metadata = as[1].metadata == nothing ? nothing : as[1].metadata
    ArrayNode(data, metadata)
end

# hcat and vcat only for ArrayNode
function Base.vcat(as::ArrayNode...)
    data = reduce(vcat, [a.data for a in as])
    metadata = as[1].metadata == nothing ? nothing : as[1].metadata
    ArrayNode(data, metadata)
end

Base.hcat(as::ArrayNode...) = reduce(catobs, collect(as))

Base.getindex(x::ArrayNode, i::VecOrRange{<:Int}) = ArrayNode(subset(x.data, i), subset(x.metadata, i))

Base.hash(e::ArrayNode, h::UInt) = hash(e.data, e.metadata), h)
(e1::ArrayNode == e2::ArrayNode) = isequal(e1.data == e2.data, true) && e1.metadata == e2.metadata
isequal(e1::ArrayNode, e2::ArrayNode) = isequal(e1.data, e2.data) && isequal(e1.metadata, e2.metadata)
