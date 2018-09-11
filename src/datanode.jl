using LearnBase
using DataFrames

abstract type AbstractNode end
abstract type AbstractBagNode{T <: AbstractNode, C} <: AbstractNode end
abstract type AbstractTreeNode{N, C} <: AbstractNode end

mutable struct ArrayNode{A,C} <: AbstractNode
	data::A
	metadata::C
end

mutable struct BagNode{T, C} <: AbstractBagNode{T, C}
	data::T
	bags::Bags
	metadata::C

	function BagNode{T, C}(data::T, bags::Union{Bags, Vector}, metadata::C) where {T <: AbstractNode, C}
		new(data, bag(bags), metadata)
	end
end

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

mutable struct TreeNode{N} <: AbstractTreeNode{N, Void}
	data::NTuple{N, AbstractNode}

	function TreeNode{N}(data::NTuple{N, AbstractNode}) where N
		assert(length(data) >= 1 && all(x -> nobs(x) == nobs(data[1]), data))
		new(data)
	end
end

ArrayNode(data::AbstractArray) = ArrayNode(data, nothing)
ArrayNode(data::AbstractNode, metadata = nothing) = data
BagNode(x::T, b::Union{Bags, Vector}, metadata::C=nothing) where {T <: AbstractNode, C} =
	BagNode{T, C}(x, b, metadata)
WeightedBagNode(x::T, b::Union{Bags, Vector}, weights::Vector{W}, metadata::C=nothing) where {T <: AbstractNode, W, C} =
	WeightedBagNode{T, W, C}(x, b, weights, metadata)
TreeNode(data::NTuple{N, AbstractNode}) where N = TreeNode{N}(data)

################################################################################
mapdata(f, x::ArrayNode) = ArrayNode(f(x.data), x.metadata)
mapdata(f, x::BagNode) = BagNode(mapdata(f, x.data), x.bags, x.metadata)
mapdata(f, x::WeightedBagNode) = WeightedBagNode(mapdata(f, x.data), x.bags, x.weights, x.metadata)
mapdata(f, x::TreeNode) = TreeNode(map(i -> mapdata(f, i), x.data))

################################################################################

LearnBase.nobs(a::ArrayNode) = size(a.data, 2)
LearnBase.nobs(a::ArrayNode, ::Type{ObsDim.Last}) = nobs(a)
LearnBase.nobs(a::AbstractBagNode) = length(a.bags)
LearnBase.nobs(a::AbstractBagNode, ::Type{ObsDim.Last}) = nobs(a)
LearnBase.nobs(a::AbstractTreeNode) = nobs(a.data[1], ObsDim.Last)
LearnBase.nobs(a::AbstractTreeNode, ::Type{ObsDim.Last}) = nobs(a)

################################################################################

Base.vcat(as::ArrayNode...) = ArrayNode(vcat([a.data for a in as]...))
Base.hcat(as::ArrayNode...) = ArrayNode(hcat([a.data for a in as]...))

function Base.cat(a::T...) where T <: AbstractNode
	data = lastcat(map(d -> d.data, a)...)
	metadata = lastcat(Iterators.filter(i -> i != nothing, map(d -> d.metadata,a))...)
	return T(data, metadata)
end

function Base.cat(a::BagNode...)
	data = lastcat(map(d -> d.data, a)...)
	metadata = lastcat(Iterators.filter(i -> i != nothing, map(d -> d.metadata,a))...)
	bags = catbags(map(d -> d.bags, a)...)
	return BagNode(data, bags, metadata)
end

function Base.cat(a::WeightedBagNode...)
	data = lastcat(map(d -> d.data, a)...)
	metadata = lastcat(Iterators.filter(i -> i != nothing, map(d -> d.metadata,a))...)
	bags = catbags(map(d -> d.bags, a)...)
	weights = lastcat(map(d -> d.weights, a)...)
	return WeightedBagNode(data, bags, weights, metadata)
end

function Base.cat(a::TreeNode...)
	data = lastcat(map(d -> d.data, a)...)
	return TreeNode(data)
end

lastcat(a::AbstractArray...) = hcat(a...)
lastcat(a::Vector...) = vcat(a...)
lastcat(a::DataFrame...) = vcat(a...)
lastcat(a::AbstractNode...) = cat(a...)
lastcat(a::Void...) = nothing
# enforces both the same length of the tuples and their structure
lastcat(a::NTuple{N, AbstractNode}...) where N = ((cat(d...) for d in zip(a...))...,)
lastcat() = nothing

################################################################################

Base.getindex(x::T, i::VecOrRange) where T <: AbstractNode = T(subset(x.data, i), subset(x.metadata, i))

function Base.getindex(x::BagNode, i::VecOrRange)
	nb, ii = remapbag(x.bags, i)
	BagNode(subset(x.data,ii), nb, subset(x.metadata, ii))
end

function Base.getindex(x::WeightedBagNode, i::VecOrRange)
	nb, ii = remapbag(x.bags, i)
	WeightedBagNode(subset(x.data,ii), nb, subset(x.weights, ii), subset(x.metadata, ii))
end

Base.getindex(x::TreeNode, i::VecOrRange) = TreeNode(subset(x.data, i))

Base.getindex(x::AbstractNode, i::Int) = x[i:i]
MLDataPattern.getobs(x::AbstractNode, i) = x[i]
MLDataPattern.getobs(x::AbstractNode, i, ::LearnBase.ObsDim.Undefined) = x[i]
MLDataPattern.getobs(x::AbstractNode, i, ::LearnBase.ObsDim.Last) = x[i]

subset(x::AbstractArray, i) = x[:, i]
subset(x::Vector, i) = x[i]
subset(x::AbstractNode, i) = x[i]
subset(x::DataFrame, i) = x[i, :]
subset(x::Void, i) = nothing
subset(xs::Tuple, i) = tuple(map(x -> x[i], xs)...)

################################################################################

Base.show(io::IO, n::AbstractNode) = dsprint(io, n)

dsprint(io::IO, n::ArrayNode; pad=[]) =
	paddedprint(io, "ArrayNode$(size(n.data))\n")

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

function dsprint(io::IO, n::AbstractTreeNode{N}; pad=[]) where {N}
	c = COLORS[(length(pad)%length(COLORS))+1]
	paddedprint(io, "TreeNode{$N}\n", color=c)

	for i in 1:length(n.data)-1
		paddedprint(io, "  ├── ", color=c, pad=pad)
		dsprint(io, n.data[i], pad=[pad; (c, "  │   ")])
	end
	paddedprint(io, "  └── ", color=c, pad=pad)
	dsprint(io, n.data[end], pad=[pad; (c, "      ")])
end
