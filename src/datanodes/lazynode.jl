"""
    LazyNode{Name, D, C} <: AbstractMillNode

Data node storing data of type `D` in a lazy manner and optional metadata of type `C`.

Source of data or its type is specified in `Name`.

See also: [`AbstractMillNode`](@ref), [`LazyModel`](@ref), [`Mill.unpack2mill`](@ref).
"""
struct LazyNode{Name, D, C} <: AbstractMillNode
    data::D
    metadata::C
end

"""
    LazyNode([Name::Symbol], d, m=nothing)
    LazyNode{Name}(d, m=nothing)

Construct a new [`LazyNode`](@ref) with name `Name`, data `d`, and metadata `m`.

# Examples
```jldoctest
julia> LazyNode(:Codons, ["GGGCGGCGA", "CCTCGCGGG"])
LazyNode{Codons} 	# 2 obs, 98 bytes
```

See also: [`AbstractMillNode`](@ref), [`LazyModel`](@ref), [`Mill.unpack2mill`](@ref).
"""
LazyNode(Name::Symbol, d::T, m::C=nothing) where {T, C} = LazyNode{Name, T, C}(d, m)
LazyNode{Name}(d::T, m::C=nothing) where {Name, T, C} = LazyNode{Name, T, C}(d, m)

"""
    Mill.unpack2mill(x::LazyNode)

Return a representation of [`LazyNode`](@ref) `x` using `Mill.jl` structures. Every custom
[`LazyNode`](@ref) should have a special method as it is used in [`LazyModel`](@ref).

# Examples
```jldoctest unpack2mill; output=false
function Mill.unpack2mill(ds::LazyNode{:Sentence})
    s = split.(ds.data, " ")
    x = NGramMatrix(reduce(vcat, s))
    BagNode(ArrayNode(x), Mill.length2bags(length.(s)))
end
# output

```
```jldoctest unpack2mill
julia> LazyNode{:Sentence}(["foo bar", "baz"]) |> Mill.unpack2mill
BagNode 	# 2 obs, 120 bytes
  └── ArrayNode(2053×3 NGramMatrix with Int64 elements) 	# 3 obs, 176 bytes
```

See also: [`LazyNode`](@ref), [`LazyModel`](@ref).
"""
function unpack2mill end

Base.ndims(x::LazyNode) = Colon()
StatsBase.nobs(a::LazyNode) = length(a.data)
StatsBase.nobs(a::LazyNode, ::Type{ObsDim.Last}) = nobs(a.data)

mapdata(f, x::LazyNode{N}) where N = LazyNode{N}(mapdata(f, x.data), x.metadata)

dropmeta(x::LazyNode{N}) where N = LazyNode{N}(x.data)

function Base.reduce(::typeof(catobs), as::Vector{<: LazyNode{N}}) where N
    LazyNode{N}(reduce(catobs, data.(as)), reduce(catobs, metadata.(as)))
end

Base.getindex(x::LazyNode{N, T}, i::VecOrRange{<:Int}) where {N, T} = LazyNode{N}(subset(x.data, i))

Base.hash(e::LazyNode, h::UInt) = hash((e.data), h)
(e1::LazyNode == e2::LazyNode) = e1.data == e2.data
Base.isequal(e1::LazyNode, e2::LazyNode) = isequal(e1.data, e2.data)
