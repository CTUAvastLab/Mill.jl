using LearnBase
using DataFrames

abstract type AbstractNode end
abstract type AbstractProductNode <: AbstractNode end
abstract type AbstractBagNode <: AbstractNode end

"""
    data(x::AbstractNode)

    return data hold by the datanode
"""
data(x::AbstractNode) = x.data

"""
    metadata(x::AbstractNode)

    return metadata hold by the datanode
"""
metadata(x::AbstractNode) = x.metadata

"""
    catobs(as...)

    concatenates `as...` into a single datanode while preserving their structure
"""
function catobs end

mapdata(f, x) = f(x)

# functions to make datanodes compatible with getindex and with MLDataPattern
Base.getindex(x::T, i::BitArray{1}) where T <: AbstractNode = x[findall(i)]
Base.getindex(x::T, i::Vector{Bool}) where T <: AbstractNode = x[findall(i)]
Base.getindex(x::AbstractNode, i::Int) = x[i:i]
Base.lastindex(ds::AbstractNode) = nobs(ds)
MLDataPattern.getobs(x::AbstractNode, i) = x[i]
MLDataPattern.getobs(x::AbstractNode, i, ::LearnBase.ObsDim.Undefined) = x[i]
MLDataPattern.getobs(x::AbstractNode, i, ::LearnBase.ObsDim.Last) = x[i]

# subset of common datatypes the way we like them
subset(x::AbstractMatrix, i) = x[:, i]
subset(x::AbstractVector, i) = x[i]
subset(x::AbstractNode, i) = x[i]
subset(x::DataFrame, i) = x[i, :]
subset(::Missing, i) = missing
subset(::Nothing, i) = nothing
subset(xs::Tuple, i) = tuple(map(x -> x[i], xs)...)
subset(xs::NamedTuple, i) = (; [k => xs[k][i] for k in keys(xs)]...)

include("arraynode.jl")

StatsBase.nobs(a::AbstractBagNode) = length(a.bags)
StatsBase.nobs(a::AbstractBagNode, ::Type{ObsDim.Last}) = nobs(a)
Base.ndims(x::AbstractBagNode) = Colon()

include("bagnode.jl")
include("weighted_bagnode.jl")
include("productnode.jl")
include("lazynode.jl")

catobs(as::DataFrame...) = reduce(catobs, collect(as))
catobs(as::Maybe{ArrayNode}...) = reduce(catobs, collect(as))
catobs(as::Maybe{BagNode}...) = reduce(catobs, collect(as))
catobs(as::Maybe{WeightedBagNode}...) = reduce(catobs, collect(as))
catobs(as::Maybe{ProductNode}...) = reduce(catobs, collect(as))

# reduction of common datatypes the way we like it
reduce(::typeof(catobs), as::Vector{<:DataFrame}) = reduce(vcat, as)
reduce(::typeof(catobs), as::Vector{<:AbstractMatrix}) = reduce(hcat, as)
reduce(::typeof(catobs), as::Vector{<:AbstractVector}) = reduce(vcat, as)
reduce(::typeof(catobs), as::Vector{Missing}) = missing
reduce(::typeof(catobs), as::Vector{Nothing}) = nothing
reduce(::typeof(catobs), as::Vector{Union{Missing, Nothing}}) = nothing
function reduce(::typeof(catobs), as::Vector{Maybe{T}}) where T <: AbstractNode
    reduce(catobs, [a for a in as if !ismissing(a)])
end

Base.cat(as::AbstractNode...; dims=:) = reduce(catobs, collect(as))

function Base.show(io::IO, @nospecialize(n::AbstractNode))
    print(io, nameof(typeof(n)))
    if !get(io, :compact, false)
        _show_data(io, n)
        print(io, " with ", nobs(n), " obs")
    end
end

_show_data(io, n::LazyNode{Name}) where {Name} = print(io, "{", Name, "}")
_show_data(io, _) = print(io)
