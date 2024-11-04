"""
    WeightedBagNode{T <: Union{AbstractMillNode, Missing}, B <: AbstractBags, W, C} <: AbstractBagNode

Structure like [`BagNode`](@ref) but allows to specify weights of type `W` of each instance.

See also: [`BagNode`](@ref), [`AbstractBagNode`](@ref), [`AbstractMillNode`](@ref), [`BagModel`](@ref).
"""
struct WeightedBagNode{T <: Maybe{AbstractMillNode}, B <: AbstractBags, W, C} <: AbstractBagNode
    data::T
    bags::B
    weights::Vector{W}
    metadata::C

    function WeightedBagNode(d::T, b::B, w::Vector{W}, m::C = nothing) where {T<:Maybe{AbstractMillNode},B<:AbstractBags,W,C}
        @assert(!ismissing(d) || all(length.(b) .== 0),
            "WeightedBagNode with `missing` in data cannot have a non-empty bag")
        @assert(ismissing(d) || numobs(d) ≥ maxindex(b),
            "Bag indices range is greater than number of observations")
        new{T, B, W, C}(d, b, w, m)
    end
end

"""
    WeightedBagNode(d, b, w::Vector, m=nothing)

Construct a new [`WeightedBagNode`](@ref) with data `d`, bags `b`, vector of weights `w` and metadata `m`.

`d` is either an [`AbstractMillNode`](@ref) or `missing`. Any other type is wrapped in an [`ArrayNode`](@ref).

If `b` is an `AbstractVector`, [`Mill.bags`](@ref) is applied first.

# Examples
```jldoctest
julia> WeightedBagNode(ArrayNode(NGramMatrix(["s1", "s2"])), bags([1:2, 0:-1]), [0.2, 0.8])
WeightedBagNode  2 obs, 144 bytes
  ╰── ArrayNode(2053×2 NGramMatrix with Int64 elements)  2 obs, 108 bytes

julia> WeightedBagNode(zeros(2, 2), [1, 2], [1, 2])
WeightedBagNode  2 obs, 240 bytes
  ╰── ArrayNode(2×2 Array with Float64 elements)  2 obs, 88 bytes
```

See also: [`BagNode`](@ref), [`AbstractBagNode`](@ref), [`AbstractMillNode`](@ref), [`BagModel`](@ref).
"""
WeightedBagNode(d::Maybe{AbstractMillNode}, b::AbstractVector, weights::Vector, metadata = nothing) =
    WeightedBagNode(d, bags(b), weights, metadata)
WeightedBagNode(d, b, w, m=nothing) = WeightedBagNode(_arraynode(d), b, w, m)

Flux.@layer :ignore WeightedBagNode

mapdata(f, x::WeightedBagNode) = WeightedBagNode(mapdata(f, x.data), x.bags, x.weights, x.metadata)

dropmeta(x::WeightedBagNode) = WeightedBagNode(dropmeta(x.data), x.bags, x.weights)

function Base.getindex(x::WeightedBagNode{T, B, W}, i::VecOrRange{<:Integer}) where {T, B, W}
    nb, ii = remapbags(x.bags, i)
    emptyismissing() && isempty(ii) && return (WeightedBagNode(missing, nb, W[], nothing))
    WeightedBagNode(x.data[ii], nb, x.weights[ii], metadata_getindex(x.metadata, i))
end

function Base.reduce(::typeof(catobs), as::Vector{<:WeightedBagNode})
    WeightedBagNode(
        reduce(catobs, data.(as)),
        reduce(vcat, [n.bags for n in as]),
        reduce(vcat, [a.weights for a in as]),
        reduce(catobs, metadata.(as))
    )
end

function removeinstances(a::WeightedBagNode, mask)
    WeightedBagNode(a.data[mask], adjustbags(a.bags, mask), a.weights[mask], a.metadata)
end

Base.hash(n::WeightedBagNode, h::UInt) = hash((n.data, n.bags, n.weights, n.metadata), h)
(n1::WeightedBagNode == n2::WeightedBagNode) =
    n1.data == n2.data && n1.bags == n2.bags && n1.weights == n2.weights && n1.metadata == n2.metadata
Base.isequal(n1::WeightedBagNode, n2::WeightedBagNode) =
    isequal(n1.data, n2.data) && isequal(n1.bags, n2.bags) && isequal(n1.weights, n2.weights) && isequal(n1.metadata, n2.metadata)
