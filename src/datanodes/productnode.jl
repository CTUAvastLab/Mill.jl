"""
    ProductNode{T, C} <: AbstractProductNode

Data node representing a Cartesian product of several spaces each represented by subtree stored in iterable of type `T`. May store metadata of type `C`.

See also: [`AbstractProductNode`](@ref), [`AbstractNode`](@ref), [`ProductModel`](@ref).
"""
struct ProductNode{T, C} <: AbstractProductNode
    data::T
    metadata::C

    function ProductNode{T, C}(data::T, metadata::C) where {T, C}
        @assert(length(data) >= 1 && all(x -> nobs(x) == nobs(data[1]), data),
                "All subtrees must have an equal amount of instances!")
        new{T, C}(data, metadata)
    end
end

"""
    ProductNode(ds, m=nothing)

Construct a new [`ProductNode`](@ref) with data `ds`, and metadata `m`. `ds` should be an iterable (preferably `Tuple` or `NamedTuple`) and all its elements must contain the same number of observations.

# Examples
```jldoctest
julia> ProductNode((ArrayNode(zeros(2, 2)), ArrayNode(Flux.onehotbatch([1, 2], 1:2))))
ProductNode with 2 obs
  ├── ArrayNode(2×2 Array with Float64 elements) with 2 obs
  └── ArrayNode(2×2 OneHotArray with Bool elements) with 2 obs

julia> ProductNode((x1 = ArrayNode(NGramMatrix(["Hello", "world"])),
                    x2 = BagNode(ArrayNode([1 2; 3 4]), [1:3, 4:4])))
ProductNode with 2 obs
  ├── x1: ArrayNode(2053×2 NGramMatrix with Int64 elements) with 2 obs
  └── x2: BagNode with 2 obs
            └── ArrayNode(2×2 Array with Int64 elements) with 2 obs

julia> ProductNode((ArrayNode([1 2; 3 4]), ArrayNode([1; 3])))
ERROR: AssertionError: All subtrees must have an equal amount of instances!
[...]
```

See also: [`AbstractProductNode`](@ref), [`AbstractNode`](@ref), [`ProductModel`](@ref).
"""
ProductNode(ds::T) where {T} = ProductNode{T, Nothing}(ds, nothing)
ProductNode(ds::T, m::C) where {T, C} = ProductNode{T, C}(ds, m)

Flux.@functor ProductNode

mapdata(f, x::ProductNode) = ProductNode(map(i -> mapdata(f, i), x.data), x.metadata)

dropmeta(x::ProductNode) = ProductNode(x.data)

Base.getindex(x::ProductNode, i::Symbol) = x.data[i]
Base.keys(x::ProductNode) = keys(x.data)

_length_error() = ArgumentError("Trying to `catobs` `ProductNode`s with different subtrees") |> throw
function _check_idxs(as::Vector{<:Tuple})
    if any(length.(as) .!= length(as[1]))
        _length_error()
    end
    1:length(as[1])
end
_check_idxs(as::Vector{<:NamedTuple{K}}) where K = keys(as[1])
_check_idxs(as) = _length_error()

_cattrees(as::Vector{T}) where T = T(reduce(catobs, getindex.(as, i)) for i in _check_idxs(as))

function reduce(::typeof(catobs), as::Vector{<:ProductNode})
    ds = data.(as)
    idxs = _check_idxs(ds)
    ProductNode(_cattrees(ds), reduce(catobs, metadata.(as)))
end

Base.getindex(x::ProductNode, i::VecOrRange{<:Int}) = ProductNode(subset(x.data, i), subset(x.metadata, i))

Base.hash(e::ProductNode, h::UInt) = hash((e.data, e.metadata), h)
(e1::ProductNode == e2::ProductNode) = e1.data == e2.data && e1.metadata == e2.metadata
Base.isequal(e1::ProductNode, e2::ProductNode) = isequal(e1.data, e2.data) && isequal(e1.metadata, e2.metadata)
