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
reduce(::typeof(catobs), as::Vector{<: Union{Missing, Nothing}}) = nothing
function reduce(::typeof(catobs), as::Vector{T}) where {T <: Union{Missing, AbstractNode}}
    reduce(catobs, [a for a in as if !ismissing(a)])
end

function reduce(::typeof(catobs), as::Vector{<: Any})
    isempty(as) && return(as)
    T = mapreduce(typeof, typejoin, as)
    T == Any && @error "cannot reduce Any"
    reduce(catobs, Vector{T}(as))
end

Base.cat(as::AbstractNode...; dims = :) = reduce(catobs, collect(as))

_cattrees(as::Vector{T}) where T <: Union{Tuple, Vector}  = tuple([reduce(catobs, [a[i] for a in as]) for i in 1:length(as[1])]...)
function _cattrees(as::Vector{T}) where T <: NamedTuple
    ks = keys(as[1])
    vs = [k => reduce(catobs, [a[k] for a in as]) for k in ks]
    (;vs...)
end

mapdata(f, x) = f(x)

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
subset(xs::NamedTuple, i) = (; [k => xs[k][i] for k in keys(xs)]...)

Base.show(io::IO, ::MIME"text/plain", n::AbstractNode) = dsprint(io, n, tr=false)
Base.show(io::IO, ::T) where T <: AbstractNode = show(io, Base.typename(T))

include("arrays.jl")

# definitions needed for all types of bag nodes
_len(a::UnitRange) = max(a.stop - a.start + 1, 0)
_len(a::Vector) = length(a)
LearnBase.nobs(a::AbstractBagNode) = length(a.bags)
LearnBase.nobs(a::AbstractBagNode, ::Type{ObsDim.Last}) = nobs(a)
Base.ndims(x::AbstractBagNode) = Colon()

dsprint(io::IO, ::Missing; pad=[], s="", tr=false) = paddedprint(io, " ∅")

include("bagnode.jl")
include("weighted_bagnode.jl")

include("ngrams.jl")
include("treenode.jl")
