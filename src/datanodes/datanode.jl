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

Supertype for any data node structure representing a multi-instance learning problem.
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

"""
    catobs(ns...)

Merge multiple nodes storing samples (observations) into one suitably promoting in the process if possible.

Similar to `Base.cat` but concatenates along the abstract \"axis\" where samples are stored.

In case of repeated calls with varying number of arguments or argument types, use `reduce(catobs, [ns...])`
to save compilation time.

# Examples
```jlddoctest
julia> catobs(ArrayNode(zeros(2, 2)), ArrayNode([1 2; 3 4]))
2×4 ArrayNode{Array{Float64,2},Nothing}:
 0.0  0.0  1.0  2.0
 0.0  0.0  3.0  4.0

julia> n = ProductNode((t1=ArrayNode(randn(2, 3)), t2=BagNode(ArrayNode(randn(3, 8)), bags([1:3, 4:5, 6:8]))))
ProductNode with 3 obs
  ├── t1: ArrayNode(2×3 Array with Float64 elements) with 3 obs
  └── t2: BagNode with 3 obs
            └── ArrayNode(3×8 Array with Float64 elements) with 8 obs

julia> catobs(n[1], n[3])
ProductNode with 2 obs
  ├── t1: ArrayNode(2×2 Array with Float64 elements) with 2 obs
  └── t2: BagNode with 2 obs
            └── ArrayNode(3×6 Array with Float64 elements) with 6 obs
```

See also: [`Mill.subset`](@ref).
"""
function catobs end

"""
    subset(n, i)

Extract a subset `i` of samples (observations) stored in node `n`.

Similar to `Base.getindex` or `MLDataPattern.getobs` but defined for all `Mill.jl` compatible data as well.

# Examples
```jlddoctest
julia> Mill.subset(ArrayNode(NGramMatrix(["Hello", "world"])), 2)
2053×1 ArrayNode{NGramMatrix{String,Int64},Nothing}:
 "world"

julia> Mill.subset(BagNode(ArrayNode(randn(2, 8)), [1:2, 3:3, 4:7, 8:8]), 1:3)
BagNode with 3 obs
  └── ArrayNode(2×7 Array with Float64 elements) with 7 obs
```

See also: [`catobs`](@ref).
"""
function subset end

"""
    removeinstances(n::AbstractBagNode, mask)

Remove instances from `n` using `mask` and remap bag indices accordingly.

# Examples
```jlddoctest
julia> b1 = BagNode(ArrayNode([1 2 3; 4 5 6]), bags([1:2, 0:-1, 3:3]))
BagNode with 3 obs
  └── ArrayNode(2×3 Array with Int64 elements) with 3 obs

julia> b2 = removeinstances(b, [false, true, true])
BagNode with 3 obs
  └── ArrayNode(2×2 Array with Int64 elements) with 2 obs

julia> b2.data
2×2 ArrayNode{Array{Int64,2},Nothing}:
 2  3
 5  6

julia> b2.bags
AlignedBags{Int64}(UnitRange{Int64}[1:1, 0:-1, 2:2])
```
"""
function removeinstances end

"""
    mapdata(f, x)

Recursively apply `f` to data in all leaves of `x`.

# Examples
```jlddoctest
julia> n1 = ProductNode((a=ArrayNode(zeros(2,2)), b=ArrayNode(ones(2,2))))
ProductNode with 2 obs
  ├── a: ArrayNode(2×2 Array with Float64 elements) with 2 obs
  └── b: ArrayNode(2×2 Array with Float64 elements) with 2 obs

julia> n2 = Mill.mapdata(x -> x .+ 1, n1)
ProductNode with 2 obs
  ├── a: ArrayNode(2×2 Array with Float64 elements) with 2 obs
  └── b: ArrayNode(2×2 Array with Float64 elements) with 2 obs

julia> Mill.data(n2).a
2×2 ArrayNode{Array{Float64,2},Nothing}:
 1.0  1.0
 1.0  1.0

julia> Mill.data(n2).b
2×2 ArrayNode{Array{Float64,2},Nothing}:
 2.0  2.0
 2.0  2.0
```
"""
mapdata(f, x) = f(x)

# functions to make datanodes compatible with getindex and with MLDataPattern
Base.getindex(x::T, i::BitArray{1}) where T <: AbstractNode = x[findall(i)]
Base.getindex(x::T, i::Vector{Bool}) where T <: AbstractNode = x[findall(i)]
Base.getindex(x::AbstractNode, i::Int) = x[i:i]
Base.lastindex(ds::AbstractNode) = nobs(ds)
MLDataPattern.getobs(x::AbstractNode, i) = x[i]
MLDataPattern.getobs(x::AbstractNode, i, ::LearnBase.ObsDim.Undefined) = x[i]
MLDataPattern.getobs(x::AbstractNode, i, ::LearnBase.ObsDim.Last) = x[i]

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

catobs(as::Maybe{ArrayNode}...) = reduce(catobs, collect(as))
catobs(as::Maybe{BagNode}...) = reduce(catobs, collect(as))
catobs(as::Maybe{WeightedBagNode}...) = reduce(catobs, collect(as))
catobs(as::Maybe{ProductNode}...) = reduce(catobs, collect(as))
catobs(as::Maybe{LazyNode}...) = reduce(catobs, collect(as))

Base.cat(as::AbstractNode...; dims=:) = catobs(as...)

# reduction of common datatypes the way we like it
Base.reduce(::typeof(catobs), as::Vector{<:DataFrame}) = reduce(vcat, as)
Base.reduce(::typeof(catobs), as::Vector{<:AbstractMatrix}) = reduce(hcat, as)
Base.reduce(::typeof(catobs), as::Vector{<:AbstractVector}) = reduce(vcat, as)
Base.reduce(::typeof(catobs), as::Vector{Missing}) = missing
Base.reduce(::typeof(catobs), as::Vector{Nothing}) = nothing
Base.reduce(::typeof(catobs), as::Vector{Union{Missing, Nothing}}) = nothing

function Base.reduce(::typeof(catobs), as::Vector{Maybe{T}}) where T <: AbstractNode
    reduce(catobs, skipmissing(as) |> collect)
end

function Base.show(io::IO, @nospecialize(n::AbstractNode))
    print(io, nameof(typeof(n)))
    if !get(io, :compact, false)
        _show_data(io, n)
        print(io, " with ", nobs(n), " obs")
    end
end

_show_data(io, n::LazyNode{Name}) where {Name} = print(io, "{", Name, "}")
_show_data(io, _) = print(io)
