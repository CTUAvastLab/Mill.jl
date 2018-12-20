using LearnBase
using DataFrames
import Base: cat, vcat, hcat

abstract type AbstractNode end
abstract type AbstractBagNode{T <: AbstractNode, C} <: AbstractNode end
abstract type AbstractTreeNode{T, C} <: AbstractNode end

mutable struct ArrayNode{A,C} <: AbstractNode
    data::A
    metadata::C
end
ArrayNode(data::AbstractArray) = ArrayNode(data, nothing)
ArrayNode(data::AbstractNode) = data
ArrayNode(data::AbstractNode, metadata::Nothing) = data

Base.ndims(x::ArrayNode) = 0



mutable struct BagNode{T, C} <: AbstractBagNode{T, C}
    data::T
    bags::Bags
    metadata::C

    function BagNode{T, C}(data::T, bags::Union{Bags, Vector}, metadata::C) where {T <: AbstractNode, C}
        new(data, bag(bags), metadata)
    end
end
BagNode(x::T, b::Union{Bags, Vector}, metadata::C=nothing) where {T <: AbstractNode, C} =
BagNode{T, C}(x, b, metadata)

Base.ndims(x::BagNode) = 0



mutable struct WeightedBagNode{T, W, C} <: AbstractBagNode{T, C}
    data::T
    bags::Bags
    weights::Vector{W}
    metadata::C

    function WeightedBagNode{T, W, C}(
                                      data::T, bags::Union{Bags, Vector}, weights::Vector{W}, metadata::C) where {T <: AbstractNode, W, C}
        new(data, bag(bags), weights, metadata)
    end
end

Base.ndims(x::WeightedBagNode) = 0

mutable struct TreeNode{T,C} <: AbstractTreeNode{T, C}
    data::T
    metadata::C

    function TreeNode{T,C}(data::T, metadata::C) where {T <: NTuple{N, AbstractNode} where N, C}
        @assert length(data) >= 1 && all(x -> nobs(x) == nobs(data[1]), data)
        new(data, metadata)
    end
end

Base.ndims(x::TreeNode) = 0

WeightedBagNode(x::T, b::Union{Bags, Vector}, weights::Vector{W}, metadata::C=nothing) where {T <: AbstractNode, W, C} =
WeightedBagNode{T, W, C}(x, b, weights, metadata)
TreeNode(data::T) where {T} = TreeNode{T, Nothing}(data, nothing)
TreeNode(data::T, metadata::C) where {T, C} = TreeNode{T, C}(data, metadata)

################################################################################

mapdata(f, x::ArrayNode) = ArrayNode(f(x.data), x.metadata)
mapdata(f, x::BagNode) = BagNode(mapdata(f, x.data), x.bags, x.metadata)
mapdata(f, x::WeightedBagNode) = WeightedBagNode(mapdata(f, x.data), x.bags, x.weights, x.metadata)
mapdata(f, x::TreeNode) = TreeNode(map(i -> mapdata(f, i), x.data), x.metadata)

data(x::AbstractNode) = x.data
data(x) = x

################################################################################

LearnBase.nobs(a::ArrayNode) = size(a.data, 2)
LearnBase.nobs(a::ArrayNode, ::Type{ObsDim.Last}) = nobs(a)
LearnBase.nobs(a::AbstractBagNode) = length(a.bags)
LearnBase.nobs(a::AbstractBagNode, ::Type{ObsDim.Last}) = nobs(a)
LearnBase.nobs(a::AbstractTreeNode) = nobs(a.data[1], ObsDim.Last)
LearnBase.nobs(a::AbstractTreeNode, ::Type{ObsDim.Last}) = nobs(a)

################################################################################
#
# FIXME: this alias would better be Union{AbstractVector{T}, Tuple{Vararg{T}}}
# and method signatures should do AbstractVecOrTuple{<:T} when they want covariance,
# but that solution currently fails (see #27188 and #27224)
AbstractVecOrTuple{T} = Union{AbstractVector{<:T}, Tuple{Vararg{T}}}

# hcat and vcat only for ArrayNode
function Base.vcat(as::ArrayNode...)
    data = reduce(vcat, [a.data for a in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    ArrayNode(data, metadata)
end

Base.hcat(as::ArrayNode...) = reduce(catobs, collect(as))

function reduce(::typeof(catobs), as::Vector{T}) where {T<:ArrayNode}
    data = reduce(catobs, [x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    ArrayNode(data, metadata)
end

function reduce(::typeof(catobs), as::Vector{T}) where {T<:BagNode}
    data = reduce(catobs, [x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    bags = _catbags([d.bags for d in as])
    BagNode(data, bags, metadata)
end

function reduce(::typeof(catobs), as::Vector{T}) where {T<:WeightedBagNode}
    data = reduce(catobs, [x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    bags = _catbags([d.bags for d in as])
    weights = reduce(catobs, [d.weights for d in as])
    WeightedBagNode(data, bags, weights, metadata)
end

function reduce(::typeof(catobs), as::Vector{T}) where {T<:TreeNode}
    data = _cattuples([x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    TreeNode(data, metadata)
end

catobs(as::T...) where {T<:AbstractNode} = reduce(catobs, collect(as))
cat(a::T, b::T; dims = 0) where {T<:AbstractNode} = reduce(catobs, [a, b])

reduce(::typeof(catobs), as::Vector{T}) where {T<:AbstractMatrix} = reduce(hcat, as)
reduce(::typeof(catobs), as::Vector{T}) where {T<:AbstractVector} = reduce(vcat, as)
reduce(::typeof(catobs), as::Vector{T}) where {T<:DataFrame} = reduce(vcat, as)
reduce(::typeof(catobs), as::Vector{T}) where {T<:Nothing} = nothing

_cattuples(as::AbstractVecOrTuple{T}) where {T <: NTuple{N, AbstractNode} where N}  = tuple(map(i -> reduce(catobs, [a[i] for a in as]), 1:length(as[1]))...)


################################################################################

Base.getindex(x::T, i::VecOrRange) where T <: AbstractNode = T(subset(x.data, i), subset(x.metadata, i))
Base.getindex(x::T, i::BitArray{1}) where T <: AbstractNode = x[findall(i)]

function Base.getindex(x::BagNode, i::VecOrRange)
    nb, ii = remapbag(x.bags, i)
    BagNode(subset(x.data,ii), nb, subset(x.metadata, i))
end

removeinstances(a::BagNode, mask) = BagNode(subset(a.data, findall(mask)), adjustbags(a.bags, mask), a.metadata)
adjustbags(bags::Vector{UnitRange{Int64}}, mask::T) where {T<:Union{Vector{Bool}, BitArray{1}}} = Mill.length2bags(map(b -> sum(@view mask[b]), bags))

function Base.getindex(x::WeightedBagNode, i::VecOrRange)
    nb, ii = remapbag(x.bags, i)
    WeightedBagNode(subset(x.data,ii), nb, subset(x.weights, ii), subset(x.metadata, ii))
end

Base.getindex(x::TreeNode, i::VecOrRange) = TreeNode(subset(x.data, i), subset(x.metadata, i))

Base.getindex(x::AbstractNode, i::Int) = x[i:i]
MLDataPattern.getobs(x::AbstractNode, i) = x[i]
MLDataPattern.getobs(x::AbstractNode, i, ::LearnBase.ObsDim.Undefined) = x[i]
MLDataPattern.getobs(x::AbstractNode, i, ::LearnBase.ObsDim.Last) = x[i]

subset(x::AbstractArray, i) = x[:, i]
subset(x::Vector, i) = x[i]
subset(x::AbstractNode, i) = x[i]
subset(x::DataFrame, i) = x[i, :]
subset(x::Nothing, i) = nothing
subset(xs::Tuple, i) = tuple(map(x -> x[i], xs)...)

################################################################################

Base.show(io::IO, n::AbstractNode) = dsprint(io, n)

dsprint(io::IO, n::ArrayNode; pad=[]) = paddedprint(io, "ArrayNode$(size(n.data))\n")

function dsprint(io::IO, n::BagNode{ArrayNode}; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io,"BagNode$(size(n.data)) with $(length(n.bags)) bag(s)\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.data, pad = [pad; (c, "      ")])
end

function dsprint(io::IO, n::BagNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io,"BagNode with $(length(n.bags)) bag(s)\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.data, pad = [pad; (c, "      ")])
end

function dsprint(io::IO, n::WeightedBagNode{ArrayNode}; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "WeightedNode$(size(n.data)) with $(length(n.bags)) bag(s) and weights Σw = $(sum(n.weights))\n", color=c)
end

function dsprint(io::IO, n::WeightedBagNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "WeightedNode with $(length(n.bags)) bag(s) and weights Σw = $(sum(n.weights))\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.data, pad = [pad; (c, "      ")])
end

function dsprint(io::IO, n::AbstractTreeNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "TreeNode{$(length(n.data))}\n", color=c)

    for i in 1:length(n.data)-1
        paddedprint(io, "  ├── ", color=c, pad=pad)
        dsprint(io, n.data[i], pad=[pad; (c, "  │   ")])
    end
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.data[end], pad=[pad; (c, "      ")])
end
