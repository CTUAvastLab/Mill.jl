using LearnBase
using DataFrames

abstract type AbstractNode{C} end
abstract type AbstractBagNode{C} <: AbstractNode{C} end
abstract type AbstractTreeNode{N, C} <: AbstractNode{C} end

mutable struct MatrixNode{C} <: AbstractNode{C}
	data::AbstractMatrix
	metadata::C
end

mutable struct BagNode{C} <: AbstractBagNode{C}
	data::AbstractNode
	bags::Bags
	metadata::C
end

mutable struct WeightedBagNode{W, C} <: AbstractBagNode{C}
	data::AbstractNode
	bags::Bags
	weights::Vector{W}
	metadata::C
end

mutable struct TreeNode{N, C} <: AbstractTreeNode{N, C}
	data::NTuple{N, AbstractNode}
	metadata::C
	function TreeNode{N, C}(data::NTuple{N, AbstractNode}, metadata::C) where {N, C}
		assert(length(data) >= 1 && all(x -> nobs(x) == nobs(data[1]), data))
		new(data, metadata)
	end
end

MatrixNode(data::AbstractMatrix) = MatrixNode(data, nothing)
BagNode(data::AbstractNode, b, metadata=nothing) = BagNode(data, bag(b), metadata)
WeightedBagNode(data::AbstractNode, b, weights::Vector, metadata=nothing) = WeightedBagNode(data, bag(b), weights, metadata)
TreeNode(data::NTuple{N, AbstractNode}, metadata::C) where {N, C} = TreeNode{N, C}(data, metadata)
TreeNode(data) = TreeNode(data, nothing)

Base.show(io::IO, n::MatrixNode, offset::Int=0) =
	paddedprint(io,"MatrixNode$(size(n.data))\n",offset)
function Base.show(io::IO, n::BagNode, offset::Int=0)
	paddedprint(io,"BagNode with $(length(n.bags)) bag(s)\n",offset)
	Base.show(io, n.data, offset+2)
end
Base.show(io::IO, n::WeightedBagNode, offset::Int=0) =
	paddedprint(io,"WeightedNode$(size(n.data)) with $(length(n.bags)) bag(s) and weights summing to $(sum(n.weights))\n",offset)
function Base.show(io::IO, n::AbstractTreeNode{N}, offset::Int=0) where {N}
	paddedprint(io,"TreeNode{$N}(\n", offset)
	foreach(m -> Base.show(io, m, offset+2), n.data)
	paddedprint(io,")\n", offset)
end

mapdata(f, x::MatrixNode) = MatrixNode(f(x.data), x.metadata)
mapdata(f, x::BagNode) = BagNode(mapdata(f, x.data), x.bags, x.metadata)
mapdata(f, x::WeightedBagNode) = WeightedBagNode(mapdata(f, x.data), x.bags, x.weights, x.metadata)
mapdata(f, x::TreeNode) = TreeNode(map(i -> mapdata(f, i), x.data), x.metadata)

LearnBase.nobs(a::MatrixNode) = size(a.data,2)
LearnBase.nobs(a::MatrixNode, ::Type{ObsDim.Last}) = nobs(a)
LearnBase.nobs(a::AbstractBagNode) = length(a.bags)
LearnBase.nobs(a::AbstractBagNode, ::Type{ObsDim.Last}) = nobs(a)
LearnBase.nobs(a::AbstractTreeNode) = nobs(a.data[1],ObsDim.Last)
LearnBase.nobs(a::AbstractTreeNode, ::Type{ObsDim.Last}) = nobs(a)

"""
		function Base.cat(a,b,c...) where {T<:AbstractNode}

		concatenates datasets a,b,c

```juliadoctest
a = BagNode(MatrixNode(rand(3,4)),[1:4])
b = BagNode(MatrixNode(rand(3,4)),[1:2,3:4])
cat(a,b)
```

	Internally, the functions calls package-specific function lastcat to enforce concatenations
	assuming that last dimension are observations. If you want to use AbstractNode with special datastores, you should extend it
"""
function Base.cat(a::T...) where T <: AbstractNode
	data = lastcat(map(d -> d.data, a)...)
	metadata = lastcat(Iterators.filter(i -> i!= nothing,map(d -> d.metadata,a))...)
	return T(data, metadata)
end

function Base.cat(a::BagNode...)
	data = lastcat(map(d -> d.data, a)...)
	metadata = lastcat(Iterators.filter(i -> i!= nothing,map(d -> d.metadata,a))...)
	bags = catbags(map(d -> d.bags, a)...)
	return BagNode(data, bags, metadata)
end

function Base.cat(a::WeightedBagNode...)
	data = lastcat(map(d -> d.data, a)...)
	metadata = lastcat(Iterators.filter(i -> i!= nothing,map(d -> d.metadata,a))...)
	bags = catbags(map(d -> d.bags, a)...)
	weights = lastcat(map(d -> d.weights, a)...)
	return WeightedBagNode(data, bags, weights, metadata)
end

function Base.cat(a::TreeNode...)
	data = lastcat(map(d -> d.data, a)...)
	metadata = lastcat(Iterators.filter(i -> i!= nothing,map(d -> d.metadata,a))...)
	return TreeNode(data, metadata)
end

lastcat(a::AbstractMatrix...) = hcat(a...)
lastcat(a::Vector...) = vcat(a...)
lastcat(a::DataFrame...) = vcat(a...)
lastcat(a::AbstractNode...) = cat(a...)
lastcat(a::Void...) = nothing
# enforces both the same length of the tuples and their structure
lastcat(a::NTuple{N, AbstractNode}...) where N = ((cat(d...) for d in zip(a...))...)
lastcat() = nothing

Base.getindex(x::T, i::VecOrRange) where T <: AbstractNode = T(subset(x.data, i), subset(x.metadata, i))
function Base.getindex(x::BagNode, i::VecOrRange)
	nb, ii = remapbag(x.bags, i)
	BagNode(subset(x.data,ii), nb, subset(x.metadata, ii))
end

function Base.getindex(x::WeightedBagNode, i::VecOrRange)
	nb, ii = remapbag(x.bags, i)
	WeightedBagNode(subset(x.data,ii), nb, subset(x.weights, ii), subset(x.metadata, ii))
end

Base.getindex(x::AbstractNode, i::Int) = x[i:i]
MLDataPattern.getobs(x::AbstractNode, i) = x[i]
MLDataPattern.getobs(x::AbstractNode, i, ::LearnBase.ObsDim.Undefined) = x[i]
MLDataPattern.getobs(x::AbstractNode, i, ::LearnBase.ObsDim.Last) = x[i]

subset(x::AbstractMatrix, i) = x[:, i]
subset(x::Vector, i) = x[i]
subset(x::AbstractNode, i) = x[i]
subset(x::DataFrame, i) = x[i, :]
subset(x::Void, i) = nothing
subset(xs::Tuple, i) = tuple(map(x -> x[i], xs)...)

"""
		sparsify(x,nnzrate)

		replace matrices with at most `nnzrate` fraction of non-zeros with SparseMatrixCSC

```juliadoctest
julia> x = TreeNode((
				TreeNode((
					MatrixNode(randn(5,5)),
					MatrixNode(zeros(5,5))
						)),
				MatrixNode(zeros(5,5))
				))
julia> mapdata(i -> sparsify(i,0.05),x)

```
"""
sparsify(x,nnzrate) = x
sparsify(x::Matrix,nnzrate) = (mean(x .!= 0) <nnzrate) ? sparse(x) : x
