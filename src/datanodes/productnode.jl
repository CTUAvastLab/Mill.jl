struct ProductNode{T,C} <: AbstractProductNode
    data::T
    metadata::C

    function ProductNode{T,C}(data::T, metadata::C) where {T, C}
        @assert(length(data) >= 1 && all(x -> nobs(x) == nobs(data[1]), data),
                "All subtrees must have an equal amount of instances!")
        new(data, metadata)
    end
end

ProductNode(data::T) where {T} = ProductNode{T, Nothing}(data, nothing)
ProductNode(data::T, metadata::C) where {T, C} = ProductNode{T, C}(data, metadata)

Flux.@functor ProductNode

mapdata(f, x::ProductNode) = ProductNode(map(i -> mapdata(f, i), x.data), x.metadata)

Base.getindex(x::ProductNode, i::Symbol) = x.data[i]
Base.keys(x::ProductNode) = keys(x.data)

function _cattrees(as::Vector{<:Tuple})
    @assert all(length.(as) .== length(as[1]))
    tuple([reduce(catobs, [a[i] for a in as]) for i in eachindex(as[1])]...)
end
function _cattrees(as::Vector{<:NamedTuple{K}}) where K
    (; [k => reduce(catobs, getindex.(as, k)) for k in K]...) 
end

function reduce(::typeof(catobs), as::Vector{T}) where {T <: ProductNode}
    d = _cattrees(data.(as))
    md = reduce(catobs, metadata.(as))
    ProductNode(d, md)
end

Base.getindex(x::ProductNode, i::VecOrRange{<:Int}) = ProductNode(subset(x.data, i), subset(x.metadata, i))

Base.hash(e::ProductNode, h::UInt) = hash((e.data, e.metadata), h)
(e1::ProductNode == e2::ProductNode) = e1.data == e2.data && e1.metadata == e2.metadata
Base.isequal(e1::ProductNode, e2::ProductNode) = isequal(e1.data, e2.data) && isequal(e1.metadata, e2.metadata)
