"""
    AbstractNode

Supertype for any structure representing a data node.
"""
abstract type AbstractNode end

"""
    AbstractProductNode <: AbstractNode

Supertype for any structure representing a data node implementing a Cartesian product of data in subtrees.
"""
abstract type AbstractProductNode <: AbstractNode end

"""
    AbstractBagNode <: AbstractNode

Supertype for any structure representing a data node that implements a multi-instance learning problem.
"""
abstract type AbstractBagNode <: AbstractNode end

"""
    Mill.data(n::AbstractNode)

Return data stored in node `n`.

# Examples
```jlddoctest
julia> Mill.data(ArrayNode([1 2; 3 4], "metadata"))
2×2 Array{Int64,2}:
 1  2
 3  4

julia> Mill.data(BagNode(ArrayNode([1 2; 3 4]), bags([1:3, 4:4]), "metadata"))
2×2 ArrayNode{Array{Int64,2},Nothing}:
 1  2
 3  4
```

See also: [`Mill.metadata`](@ref)
"""
data(n::AbstractNode) = n.data

"""
    Mill.metadata(n::AbstractNode)

Return metadata stored in node `n`.

# Examples
```jlddoctest
julia> Mill.metadata(ArrayNode([1 2; 3 4], "metadata"))
"metadata"

julia> Mill.metadata(BagNode(ArrayNode([1 2; 3 4]), bags([1:3, 4:4]), "metadata"))
"metadata"
```

See also: [`Mill.data`](@ref)
"""
metadata(x::AbstractNode) = x.metadata


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

StatsBase.nobs(::Missing) = nothing

include("bagnode.jl")
include("weighted_bagnode.jl")

StatsBase.nobs(a::AbstractBagNode) = length(a.bags)
StatsBase.nobs(a::AbstractBagNode, ::Type{ObsDim.Last}) = nobs(a)
Base.ndims(x::AbstractBagNode) = Colon()

include("productnode.jl")

Base.ndims(x::AbstractProductNode) = Colon()
StatsBase.nobs(a::AbstractProductNode) = nobs(a.data[1], ObsDim.Last)
StatsBase.nobs(a::AbstractProductNode, ::Type{ObsDim.Last}) = nobs(a)

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
    reduce(catobs, skipmissing(as) |> collect)
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
