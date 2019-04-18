using LearnBase
using DataFrames
import Base: cat, vcat, hcat, _cat, lastindex, getindex

abstract type AbstractNode end
abstract type AbstractTreeNode <: AbstractNode end
abstract type AbstractBagNode <: AbstractNode end


# FIXME: this alias would better be Union{AbstractVector{T}, Tuple{Vararg{T}}}
# and method signatures should do AbstractVecOrTuple{<:T} when they want covariance,
# but that solution currently fails (see #27188 and #27224)
AbstractVecOrTuple{T} = Union{AbstractVector{<:T}, Tuple{Vararg{T}}}


"""
    data(x::AbstractNode)

    return data hold by the datanode
"""
data(x::AbstractNode) = x.data
data(x) = x

"""
    catobs(as...)

    concatenates `as...` into a single datanode while preserving their structure
"""
catobs(as...) = reduce(catobs, collect(as))

# reduction of common datatypes the way we like it
reduce(::typeof(catobs), as::Vector{<: AbstractMatrix}) = reduce(hcat, as)
reduce(::typeof(catobs), as::Vector{<: AbstractVector}) = reduce(vcat, as)
reduce(::typeof(catobs), as::Vector{<: DataFrame}) = reduce(vcat, as)
reduce(::typeof(catobs), as::Vector{<: Missing}) = missing
reduce(::typeof(catobs), as::Vector{<: Nothing}) = nothing

_cattuples(as::AbstractVecOrTuple{T}) where {T <: NTuple{N, AbstractNode} where N}  = tuple(map(i -> reduce(catobs, [a[i] for a in as]), 1:length(as[1]))...)

# functions to make datanodes compatible with getindex and with MLDataPattern
Base.getindex(x::T, i::BitArray{1}) where T <: AbstractNode = x[findall(i)]
Base.getindex(x::T, i::Vector{Bool}) where T <: AbstractNode = x[findall(i)]
Base.getindex(x::AbstractNode, i::Int) = x[i:i]
Base.lastindex(ds::AbstractNode) = nobs(ds)
MLDataPattern.getobs(x::AbstractNode, i) = x[i]
MLDataPattern.getobs(x::AbstractNode, i, ::LearnBase.ObsDim.Undefined) = x[i]
MLDataPattern.getobs(x::AbstractNode, i, ::LearnBase.ObsDim.Last) = x[i]


#subset of common datatypes the way we like them
subset(x::AbstractMatrix, i) = x[:, i]
subset(x::AbstractVector, i) = x[i]
subset(x::AbstractNode, i) = x[i]
subset(x::DataFrame, i) = x[i, :]
subset(::Missing, i) = missing
subset(::Nothing, i) = nothing
subset(xs::Tuple, i) = tuple(map(x -> x[i], xs)...)

Base.show(io::IO, n::AbstractNode) = dsprint(io, n, tr=false)

include("arrays.jl")

# definitions needed for all types of bag nodes
_len(a::UnitRange) = max(a.stop - a.start + 1, 0)
_len(a::Vector) = length(a)
LearnBase.nobs(a::AbstractBagNode) = length(a.bags)
LearnBase.nobs(a::AbstractBagNode, ::Type{ObsDim.Last}) = nobs(a)
Base.ndims(x::AbstractBagNode) = Colon()
dsprint(io::IO, ::Missing; pad=[], s="", tr=false) where T = paddedprint(io, " ∅ ")

include("bagnode.jl")
include("weighted_bagnode.jl")

include("ngrams.jl")
include("treenode.jl")

Base.cat(a::AbstractNode, b::AbstractNode; dims = :) = _cat(a, b, dims)
function Base.cat(a::T, b::T, dims = Colon) where {T <: AbstractNode}
    _cat(a, b, dims)
end

_cat(a, b, ::Colon) = reduce(catobs, [a, b])

