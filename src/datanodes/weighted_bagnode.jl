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

    function WeightedBagNode(d::T, b::B, w::Vector{W}, m::C=nothing) where {T <: Maybe{AbstractMillNode}, B <: AbstractBags, W, C}
        ismissing(d) && any(length.(b.bags) .!= 0) && error("WeightedBagNode with nothing in data cannot have a non-empty bag")
        new{T, B, W, C}(d, b, w, m)
    end
end

"""
    WeightedBagNode(d::Union{AbstractMillNode, Missing}, b::AbstractBags, w::Vector, m=nothing)
    WeightedBagNode(d::Union{AbstractMillNode, Missing}, b::AbstractVector, w::Vector, m=nothing)

Construct a new [`WeightedBagNode`](@ref) with data `d`, bags `b`, weights `w` and metadata `m`. If `b` is an `AbstractVector`, [`Mill.bags`](@ref) is applied first.

# Examples
```jldoctest
julia> BagNode(ArrayNode(NGramMatrix(["s1", "s2"])), bags([1:2, 0:-1]), [0.2, 0.8])
BagNode 	# 2 obs, 184 bytes
  └── ArrayNode(2053×2 NGramMatrix with Int64 elements) 	# 2 obs, 140 bytes

julia> BagNode(ArrayNode(zeros(2, 2)), [1, 2], [1, 2])
BagNode 	# 2 obs, 160 bytes
  └── ArrayNode(2×2 Array with Float64 elements) 	# 2 obs, 80 bytes
```

See also: [`BagNode`](@ref), [`AbstractBagNode`](@ref), [`AbstractMillNode`](@ref), [`BagModel`](@ref).
"""
WeightedBagNode(d::Maybe{AbstractMillNode}, b::AbstractVector, weights::Vector, metadata=nothing) =
    WeightedBagNode(d, bags(b), weights, metadata)


Flux.@functor WeightedBagNode

mapdata(f, x::WeightedBagNode) = WeightedBagNode(mapdata(f, x.data), x.bags, x.weights, x.metadata)

dropmeta(x::WeightedBagNode) = WeightedBagNode(dropmeta(x.data), x.bags, x.weights)

function Base.getindex(x::WeightedBagNode{T, B, W}, i::VecOrRange{<:Int}) where {T, B, W}
    nb, ii = remapbags(x.bags, i)
    emptyismissing() && isempty(ii) && return(WeightedBagNode(missing, nb, W[], nothing))
    WeightedBagNode(subset(x.data,ii), nb, subset(x.weights, ii), subset(x.metadata, ii))
end

function reduce(::typeof(catobs), as::Vector{<:WeightedBagNode})
    d = filter(!ismissing, data.(as))
    md = filter(!isnothing, metadata.(as))
    bags = reduce(vcat, [n.bags for n in as])
    WeightedBagNode(reduce(catobs, d), bags, reduce(vcat, [a.weights for a in as]), reduce(catobs, md))
end

function removeinstances(a::WeightedBagNode, mask)
    WeightedBagNode(subset(a.data, findall(mask)),
        adjustbags(a.bags, mask),
        subset(a.weights, findall(mask)), a.metadata)
end

Base.hash(e::WeightedBagNode, h::UInt) = hash((e.data, e.bags, e.weights, e.metadata), h)
(e1::WeightedBagNode == e2::WeightedBagNode) =
    e1.data == e2.data && e1.bags == e2.bags && e1.weights == e2.weights && e1.metadata == e2.metadata
Base.isequal(e1::WeightedBagNode, e2::WeightedBagNode) =
    isequal(e1.data, e2.data) && isequal(e1.bags, e2.bags) && isequal(e1.weights, e2.weights) && isequal(e1.metadata, e2.metadata)
