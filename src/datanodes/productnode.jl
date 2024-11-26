"""
    ProductNode{T, C} <: AbstractProductNode

Data node representing a Cartesian product of several spaces each represented by subtree stored
in iterable of type `T`. May store metadata of type `C`.

See also: [`AbstractProductNode`](@ref), [`AbstractMillNode`](@ref), [`ProductModel`](@ref).
"""
struct ProductNode{T, C} <: AbstractProductNode
    data::T
    metadata::C

    function ProductNode(data::Union{Tuple, NamedTuple, AbstractVector}, metadata=nothing)
        @assert !isempty(data) "Provide at least one subtree!"
        data = map(_arraynode, data)
        l = numobs(data[1])
        @assert all(n -> numobs(n) == l, data) "All subtrees must have an equal amount of instances!"
        new{typeof(data), typeof(metadata)}(data, metadata)
    end
end

"""
    ProductNode(dss, m=nothing)
    ProductNode(m=nothing; dss...)

Construct a new [`ProductNode`](@ref) with data `dss`, and metadata `m`.

`dss` should be a `Tuple` or `NamedTuple` and all its elements must contain
the same number of observations.

If any element of `dss` is not an [`AbstractMillNode`](@ref) it is first wrapped in
an [`ArrayNode`](@ref).

# Examples
```jldoctest
julia> ProductNode((ArrayNode(zeros(2, 2)), ArrayNode(Flux.onehotbatch([1, 2], 1:2))))
ProductNode  2 obs
  ├── ArrayNode(2×2 Array with Float64 elements)  2 obs
  ╰── ArrayNode(2×2 OneHotArray with Bool elements)  2 obs

julia> ProductNode(x1 = ArrayNode(NGramMatrix(["Hello", "world"])),
                   x2 = BagNode(ArrayNode([1 2; 3 4]), [1:2, 0:-1]))
ProductNode  2 obs
  ├── x1: ArrayNode(2053×2 NGramMatrix with Int64 elements)  2 obs
  ╰── x2: BagNode  2 obs
            ╰── ArrayNode(2×2 Array with Int64 elements)  2 obs

julia> ProductNode([1 2 3])
ProductNode  3 obs
  ╰── ArrayNode(1×3 Array with Int64 elements)  3 obs

julia> ProductNode((ArrayNode([1 2; 3 4]), ArrayNode([1 2 3; 4 5 6])))
ERROR: AssertionError: All subtrees must have an equal amount of instances!
[...]
```

See also: [`AbstractProductNode`](@ref), [`AbstractMillNode`](@ref), [`ProductModel`](@ref).
"""
ProductNode(ds, args...) = ProductNode(tuple(ds), args...)
ProductNode(args...; ns...) = ProductNode(NamedTuple(ns), args...)

Flux.@layer :ignore ProductNode

mapdata(f, x::ProductNode) = ProductNode(map(i -> mapdata(f, i), x.data), x.metadata)

dropmeta(x::ProductNode) = ProductNode(map(dropmeta, x.data))

Base.getindex(x::ProductNode, i::Symbol) = x.data[i]
Base.keys(x::ProductNode) = keys(x.data)
Base.haskey(x::ProductNode{<:NamedTuple}, k::Symbol) = haskey(x.data, k)

_check_idxs(as::Vector{<:Union{Vector, Tuple}}) = all(isequal(length(as[1])), (length(a) for a in as))
_check_idxs(as::Vector{<:NamedTuple{K}}) where K = true
_check_idxs(as) = false

_cattrees(as::Vector{<:Vector}) = [reduce(catobs, [a[i] for a in as]) for i in eachindex(as[1])]
_cattrees(as::Vector{T}) where T <: Tuple = T(reduce(catobs, [a[i] for a in as]) for i in eachindex(as[1]))
_cattrees(as::Vector{T}) where T <: NamedTuple = T(reduce(catobs, [a[i] for a in as]) for i in keys(as[1]))
@generated function _cattrees(xs::Vector{NamedTuple{K, T}}) where {K, T}
    chs = map(K, T.parameters) do k, t
        :(reduce(catobs, $t[x.$k for x in xs]))
    end
    quote
        NamedTuple{$K}(tuple($(chs...)))
    end
end

function Base.reduce(::typeof(catobs), as::Vector{<:ProductNode})
    ds = data.(as)
    if !_check_idxs(ds)
        ArgumentError("Trying to `catobs` `ProductNode`s with different subtrees") |> throw
    end
    ProductNode(_cattrees(ds), reduce(catobs, metadata.(as)))
end

function Base.getindex(x::ProductNode, i::VecOrRange{<:Integer})
    ProductNode(map(v -> v[i], x.data), metadata_getindex(x, i))
end

Base.hash(n::ProductNode, h::UInt) = hash((n.data, n.metadata), h)
(n1::ProductNode == n2::ProductNode) = n1.data == n2.data && n1.metadata == n2.metadata
Base.isequal(n1::ProductNode, n2::ProductNode) = isequal(n1.data, n2.data) && isequal(n1.metadata, n2.metadata)
