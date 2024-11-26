"""
    ArrayNode{A <: AbstractArray, C} <: AbstractMillNode

Data node for storing array-like data of type `A` and metadata of type `C`. The convention is that
samples are stored along the last axis, e.g. in columns of a matrix.

See also: [`AbstractMillNode`](@ref), [`ArrayModel`](@ref).
"""
struct ArrayNode{A<:AbstractArray, C} <: AbstractMillNode
    data::A
    metadata::C
end

"""
    ArrayNode(d::AbstractArray, m=nothing)

Construct a new [`ArrayNode`](@ref) with data `d` and metadata `m`.

# Examples
```jldoctest
julia> a = ArrayNode([1 2; 3 4; 5 6])
3×2 ArrayNode{Matrix{Int64}, Nothing}:
 1  2
 3  4
 5  6
```

See also: [`AbstractMillNode`](@ref), [`ArrayModel`](@ref).
"""
ArrayNode(d::AbstractArray) = ArrayNode(d, nothing)

Flux.@layer :ignore ArrayNode

mapdata(f, x::ArrayNode) = ArrayNode(mapdata(f, x.data), x.metadata)

dropmeta(x::ArrayNode) = ArrayNode(x.data)

Base.size(x::ArrayNode) = size(x.data)

function Base.reduce(::typeof(catobs), as::Vector{<:ArrayNode})
    ArrayNode(reduce(catobs, data.(as)), reduce(catobs, metadata.(as)))
end

_cat_meta(_, ::Vector{Nothing}) = nothing
_cat_meta(f, m) = reduce(f, m)

Base.vcat(as::ArrayNode...) = reduce(vcat, collect(as))
function Base.reduce(::typeof(vcat), as::Vector{<:ArrayNode})
    ArrayNode(reduce(vcat, data.(as)), _cat_meta(vcat, metadata.(as)))
end

Base.hcat(as::ArrayNode...) = reduce(hcat, collect(as))
function Base.reduce(::typeof(hcat), as::Vector{<:ArrayNode})
    ArrayNode(reduce(hcat, data.(as)), _cat_meta(hcat, metadata.(as)))
end

function Base.getindex(x::ArrayNode, i::VecOrRange{<:Integer})
    ArrayNode(x.data[:, i], metadata_getindex(x, i))
end

_arraynode(m) = ArrayNode(m)
_arraynode(m::AbstractMillNode) = m

Base.hash(n::ArrayNode, h::UInt) = hash((n.data, n.metadata), h)
(n1::ArrayNode == n2::ArrayNode) = isequal(n1.data == n2.data, true) && n1.metadata == n2.metadata
Base.isequal(n1::ArrayNode, n2::ArrayNode) = isequal(n1.data, n2.data) && isequal(n1.metadata, n2.metadata)
