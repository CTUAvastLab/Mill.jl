"""
    AbstractMillNode

Supertype for any structure representing a data node.
"""
abstract type AbstractMillNode end

"""
    AbstractProductNode <: AbstractMillNode

Supertype for any structure representing a data node implementing a Cartesian product of data in subtrees.
"""
abstract type AbstractProductNode <: AbstractMillNode end

"""
    AbstractBagNode <: AbstractMillNode

Supertype for any data node structure representing a multi-instance learning problem.
"""
abstract type AbstractBagNode <: AbstractMillNode end

"""
    Mill.data(n::AbstractMillNode)

Return data stored in node `n`.

# Examples
```jldoctest
julia> Mill.data(ArrayNode([1 2; 3 4], "metadata"))
2×2 Matrix{Int64}:
 1  2
 3  4

julia> Mill.data(BagNode(ArrayNode([1 2; 3 4]), [1, 2], "metadata"))
2×2 ArrayNode{Matrix{Int64}, Nothing}:
 1  2
 3  4
```

See also: [`Mill.metadata`](@ref)
"""
data(n::AbstractMillNode) = n.data

"""
    Mill.metadata(n::AbstractMillNode)

Return metadata stored in node `n`.

# Examples
```jldoctest
julia> Mill.metadata(ArrayNode([1 2; 3 4], ["foo", "bar"]))
2-element Vector{String}:
 "foo"
 "bar"

julia> Mill.metadata(BagNode(ArrayNode([1 2; 3 4]), [1, 2], ["metadata"]))
1-element Vector{String}:
 "metadata"
```

See also: [`Mill.data`](@ref), [`Mill.dropmeta`](@ref), [`Mill.metadata_getindex`](@ref).
"""
metadata(x::AbstractMillNode) = x.metadata

"""
    catobs(ns...)

Merge multiple nodes storing samples (observations) into one suitably promoting in the process if possible.

Similar to `Base.cat` but concatenates along the abstract \"axis\" where samples are stored.

In case of repeated calls with varying number of arguments or argument types, use `reduce(catobs, [ns...])`
to save compilation time.

# Examples
```jldoctest
julia> catobs(ArrayNode(zeros(2, 2)), ArrayNode([1 2; 3 4]))
2×4 ArrayNode{Matrix{Float64}, Nothing}:
 0.0  0.0  1.0  2.0
 0.0  0.0  3.0  4.0

julia> n = ProductNode(t1=ArrayNode(randn(2, 3)), t2=BagNode(ArrayNode(randn(3, 8)), bags([1:3, 4:5, 6:8])))
ProductNode  3 obs, 0 bytes
  ├── t1: ArrayNode(2×3 Array with Float64 elements)  3 obs, 104 bytes
  ╰── t2: BagNode  3 obs, 96 bytes
            ╰── ArrayNode(3×8 Array with Float64 elements)  8 obs, 248 bytes

julia> catobs(n[1], n[3])
ProductNode  2 obs, 0 bytes
  ├── t1: ArrayNode(2×2 Array with Float64 elements)  2 obs, 88 bytes
  ╰── t2: BagNode  2 obs, 80 bytes
            ╰── ArrayNode(3×6 Array with Float64 elements)  6 obs, 200 bytes
```
"""
function catobs end

"""
    removeinstances(n::AbstractBagNode, mask)

Remove instances from `n` using `mask` and remap bag indices accordingly.

# Examples
```jldoctest
julia> b1 = BagNode(ArrayNode([1 2 3; 4 5 6]), bags([1:2, 0:-1, 3:3]))
BagNode  3 obs, 96 bytes
  ╰── ArrayNode(2×3 Array with Int64 elements)  3 obs, 104 bytes

julia> b2 = removeinstances(b1, [false, true, true])
BagNode  3 obs, 96 bytes
  ╰── ArrayNode(2×2 Array with Int64 elements)  2 obs, 88 bytes

julia> b2.data
2×2 ArrayNode{Matrix{Int64}, Nothing}:
 2  3
 5  6

julia> b2.bags
AlignedBags{Int64}(UnitRange{Int64}[1:1, 0:-1, 2:2])
```
"""
function removeinstances end

"""
    dropmeta(n:AbstractMillNode)

Drop metadata stored in data node `n` (recursively).

# Examples
```jldoctest
julia> n1 = ArrayNode(NGramMatrix(["foo", "bar"]), ["metafoo", "metabar"])
2053×2 ArrayNode{NGramMatrix{String, Vector{String}, Int64}, Vector{String}}:
 "foo"
 "bar"

julia> n2 = dropmeta(n1)
2053×2 ArrayNode{NGramMatrix{String, Vector{String}, Int64}, Nothing}:
 "foo"
 "bar"

julia> isnothing(Mill.metadata(n2))
true
```

See also: [`Mill.metadata`](@ref), [`Mill.metadata_getindex`](@ref).
"""
function dropmeta end

"""
    mapdata(f, x)

Recursively apply `f` to data in all leaves of `x`.

# Examples
```jldoctest
julia> n1 = ProductNode(a=zeros(2,2), b=ones(2,2))
ProductNode  2 obs, 0 bytes
  ├── a: ArrayNode(2×2 Array with Float64 elements)  2 obs, 88 bytes
  ╰── b: ArrayNode(2×2 Array with Float64 elements)  2 obs, 88 bytes

julia> n2 = Mill.mapdata(x -> x .+ 1, n1)
ProductNode  2 obs, 0 bytes
  ├── a: ArrayNode(2×2 Array with Float64 elements)  2 obs, 88 bytes
  ╰── b: ArrayNode(2×2 Array with Float64 elements)  2 obs, 88 bytes

julia> Mill.data(n2).a
2×2 ArrayNode{Matrix{Float64}, Nothing}:
 1.0  1.0
 1.0  1.0

julia> Mill.data(n2).b
2×2 ArrayNode{Matrix{Float64}, Nothing}:
 2.0  2.0
 2.0  2.0
```
"""
mapdata(f, x) = f(x)

Base.getindex(x::AbstractMillNode, i::Integer) = getindex(x, i:i)
Base.lastindex(ds::AbstractMillNode) = numobs(ds)
MLUtils.getobs(x::AbstractMillNode) = x
MLUtils.getobs(x::AbstractMillNode, i) = x[i]
MLUtils.unbatch(x::AbstractMillNode) = [x[i] for i in 1:numobs(x)]

"""
    metadata_getindex(x, i::Integer)
    metadata_getindex(x, i::VecOrRange{<:Integer})

Index into metadata `x`. In [`Mill.jl`](@ref), it is assumed that the second or last dimension
indexes into observations, whichever is smaller. This function can be used when implementing
custom subtypes of [`AbstractMillNode`](@ref).

# Examples
```jldoctest
julia> Mill.metadata_getindex(["foo", "bar", "baz"], 2)
"bar"

julia> Mill.metadata_getindex(["foo", "bar", "baz"], 2:3)
2-element Vector{String}:
 "bar"
 "baz"

julia> Mill.metadata_getindex([1 2 3; 4 5 6], 2)
2-element Vector{Int64}:
 2
 5

julia> Mill.metadata_getindex([1 2 3; 4 5 6], [1, 3])
2×2 Matrix{Int64}:
 1  3
 4  6
```

See also: [`Mill.metadata`](@ref), [`Mill.dropmeta`](@ref).
"""
metadata_getindex(x, _) = x
@generated function metadata_getindex(x::AbstractArray{T, U}, i) where {T, U}
    U == 1 ? :(getindex(x, i)) : :(getindex(x, :, i, $(Colon() for _ in 3:U)...))
end

include("arraynode.jl")
Base.ndims(::ArrayNode) = Colon()
MLUtils.numobs(x::ArrayNode) = numobs(x.data)

include("bagnode.jl")
include("weighted_bagnode.jl")
Base.ndims(::AbstractBagNode) = Colon()
MLUtils.numobs(x::AbstractBagNode) = length(x.bags)

include("productnode.jl")
Base.ndims(::AbstractProductNode) = Colon()
MLUtils.numobs(x::AbstractProductNode) = numobs(x.data[1])

include("lazynode.jl")
Base.ndims(::LazyNode) = Colon()
MLUtils.numobs(x::LazyNode) = numobs(x.data)

catobs(as::Maybe{ArrayNode}...) = reduce(catobs, collect(as))
catobs(as::Maybe{BagNode}...) = reduce(catobs, collect(as))
catobs(as::Maybe{WeightedBagNode}...) = reduce(catobs, collect(as))
catobs(as::Maybe{ProductNode}...) = reduce(catobs, collect(as))
catobs(as::Maybe{LazyNode}...) = reduce(catobs, collect(as))

catobs(as::AbstractVector{<:AbstractMillNode}) = reduce(catobs, as)
MLUtils.batch(xs::AbstractVecOrTuple{<:AbstractMillNode}) = reduce(catobs, xs)

Base.cat(as::AbstractMillNode...; dims::Colon) = catobs(as...)

Base.reduce(::typeof(catobs), as::Vector{<:DataFrame}) = reduce(vcat, as)
Base.reduce(::typeof(catobs), as::Vector{<:AbstractMatrix}) = reduce(hcat, as)
Base.reduce(::typeof(catobs), as::Vector{<:AbstractVector}) = reduce(vcat, as)
Base.reduce(::typeof(catobs), as::Vector{Missing}) = missing
Base.reduce(::typeof(catobs), as::Vector{Nothing}) = nothing
Base.reduce(::typeof(catobs), as::Vector{Union{Missing, Nothing}}) = nothing

function Base.reduce(::typeof(catobs), as::Vector{Maybe{T}}) where T <: Union{
                                                                            AbstractMillNode,
                                                                            AbstractVector,
                                                                            AbstractMatrix
                                                                        }
    reduce(catobs, skipmissing(as))
end

function Base.reduce(::typeof(catobs), as::Vector{Optional{T}}) where T <: Union{
                                                                            AbstractMillNode,
                                                                            AbstractVector,
                                                                            AbstractMatrix
                                                                        }
    reduce(catobs, [a for a in as if !isnothing(a)])
end
