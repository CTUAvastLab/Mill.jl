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
    ArrayNode(reduce(catobs, data.(as)), reduce(catobs, metadata.(as)))
end

_cat_meta(f, m::Vector{Nothing}) = nothing
_cat_meta(f, m) = reduce(f, m)

Base.vcat(as::ArrayNode...) = reduce(vcat, collect(as))
function Base.reduce(::typeof(vcat), as::Vector{<:ArrayNode})
    ArrayNode(reduce(vcat, data.(as)), _cat_meta(vcat, metadata.(as)))
end

Base.hcat(as::ArrayNode...) = reduce(hcat, collect(as))
function Base.reduce(::typeof(hcat), as::Vector{<:ArrayNode})
    ArrayNode(reduce(hcat, data.(as)), _cat_meta(hcat, metadata.(as)))
end

Base.getindex(x::ArrayNode, i::VecOrRange{<:Int}) = ArrayNode(subset(x.data, i), subset(x.metadata, i))

Base.hash(e::ArrayNode, h::UInt) = hash((e.data, e.metadata), h)
(e1::ArrayNode == e2::ArrayNode) = isequal(e1.data == e2.data, true) && e1.metadata == e2.metadata
Base.isequal(e1::ArrayNode, e2::ArrayNode) = isequal(e1.data, e2.data) && isequal(e1.metadata, e2.metadata)

function Base.show(io::IO, ::MIME"text/plain", @nospecialize(n::ArrayNode))
    print(io, join(size(n.data), "×"), " ", summary(n))
    if !isempty(n.data)
        print(io, ":\n")
        Base.print_array(IOContext(io, :typeinfo => eltype(n.data)), n.data)
    end
end

function _show_data(io, n::ArrayNode{T}) where T <: AbstractArray
    print(io, "(")
    if ndims(n.data) == 1
        print(io, nameof(T), " of length ", length(n.data))
    else
        print(io, join(size(n.data), "×"), " ", nameof(T))
    end
    print(io, " with ", eltype(n.data), " elements)")
end
