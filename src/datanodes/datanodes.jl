using LearnBase
using DataFrames
import Base: cat, vcat, hcat

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
    catobs(as::T...)

    concatenates `as...` into a single datanode while preserving their structure
"""
catobs(as::T...) where {T<:AbstractNode} = reduce(catobs, collect(as))
cat(a::T, b::T; dims = 0) where {T<:AbstractNode} = reduce(catobs, [a, b])

# reduction of common datatypes the way we like it
reduce(::typeof(catobs), as::Vector{T}) where {T<:AbstractMatrix} = reduce(hcat, as)
reduce(::typeof(catobs), as::Vector{T}) where {T<:AbstractVector} = reduce(vcat, as)
reduce(::typeof(catobs), as::Vector{T}) where {T<:DataFrame} = reduce(vcat, as)
reduce(::typeof(catobs), as::Vector{T}) where {T<:Nothing} = nothing

_cattuples(as::AbstractVecOrTuple{T}) where {T <: NTuple{N, AbstractNode} where N}  = tuple(map(i -> reduce(catobs, [a[i] for a in as]), 1:length(as[1]))...)


# functions to make datanodes compatible with getindex and with MLDataPattern
Base.getindex(x::T, i::BitArray{1}) where T <: AbstractNode = x[findall(i)]
Base.getindex(x::T, i::Vector{Bool}) where T <: AbstractNode = x[findall(i)]
Base.getindex(x::AbstractNode, i::Int) = x[i:i]
MLDataPattern.getobs(x::AbstractNode, i) = x[i]
MLDataPattern.getobs(x::AbstractNode, i, ::LearnBase.ObsDim.Undefined) = x[i]
MLDataPattern.getobs(x::AbstractNode, i, ::LearnBase.ObsDim.Last) = x[i]

#subset of common datatypes the way we like them
subset(x::AbstractMatrix, i) = x[:, i]
subset(x::AbstractVector, i) = x[i]
subset(x::AbstractNode, i) = x[i]
subset(x::DataFrame, i) = x[i, :]
subset(x::Nothing, i) = nothing
subset(xs::Tuple, i) = tuple(map(x -> x[i], xs)...)


Base.show(io::IO, n::AbstractNode) = dsprint(io, n)

include("arrays.jl")
include("bagnode.jl")
include("weighted_bagnode.jl")
include("ngrams.jl")
include("treenode.jl")

