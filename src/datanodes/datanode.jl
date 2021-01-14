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
data(x) = x

"""
    catobs(as...)

    concatenates `as...` into a single datanode while preserving their structure
"""
catobs(as...) = reduce(catobs, collect(as))

# reduction of common datatypes the way we like it
reduce(::typeof(catobs), as::Vector{<:AbstractMatrix}) = reduce(hcat, as)
reduce(::typeof(catobs), as::Vector{<:AbstractVector}) = reduce(vcat, as)
Zygote.@adjoint function reduce(::typeof(catobs), as::Vector{<:AbstractMatrix})
  sz = cumsum(size.(as, 2))
  return reduce(hcat, as), Δ -> (nothing, map(n -> Zygote.pull_block_horz(sz[n], Δ, as[n]), eachindex(as)))
end

reduce(::typeof(catobs), as::Vector{<:DataFrame}) = reduce(vcat, as)
reduce(::typeof(catobs), as::Vector{Missing}) = missing
reduce(::typeof(catobs), as::Vector{Nothing}) = nothing
reduce(::typeof(catobs), as::Vector{Union{Missing, Nothing}}) = nothing
reduce(::typeof(catobs), as::Vector{<:Maybe{AbstractNode}}) = reduce(catobs, [a for a in as if !ismissing(a)])

function reduce(::typeof(catobs), as::Vector{<:Any})
    isempty(as) && return(as)
    T = mapreduce(typeof, typejoin, as)
    T === Any && @error "cannot reduce Any"
    reduce(catobs, Vector{T}(as))
end

Base.cat(as::AbstractNode...; dims = :) = reduce(catobs, collect(as))

_cattrees(as::Vector{T}) where T <: Union{Tuple, Vector}  = tuple([reduce(catobs, [a[i] for a in as]) for i in 1:length(as[1])]...)
function _cattrees(as::Vector{T}) where T <: NamedTuple
    ks = keys(as[1])
    vs = [k => reduce(catobs, [a[k] for a in as]) for k in ks]
    (; vs...)
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

function Base.show(io::IO, ::MIME"text/plain", @nospecialize n::ArrayNode)
    print(io, join(size(n.data), "×"), " ", summary(n))
    if !(isempty(n.data))
        print(io, ":\n")
        print_array(IOContext(io, :typeinfo => eltype(n.data)), n.data)
    end
end

function Base.show(io::IO, @nospecialize(n::AbstractNode))
    print(io, nameof(typeof(n)))
    if !get(io, :compact, false)
        _show_data(io, n)
        print(io, " with ", nobs(n), " obs")
    end
end

function _show_data(io, n::ArrayNode{T}) where T <: AbstractArray
    print(io, "(")
    if ndims(n.data) == 1
        print(io, nameof(T), " of length ", length(n.data))
    else
        print(io, join(size(n.data), "×"), " ", nameof(T))
    end
    print(io, ", ", eltype(n.data), ")")
end

_show_data(io, n::LazyNode{Name}) where {Name} = print(io, "{", Name, "}")
_show_data(io, _) = print(io)
