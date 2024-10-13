"""
    LazyNode{Name, D, C} <: AbstractMillNode

Data node storing data of type `D` in a lazy manner and optional metadata of type `C`.

Source of data or its type is specified in `Name`.

See also: [`AbstractMillNode`](@ref), [`LazyModel`](@ref), [`Mill.unpack2mill`](@ref).
"""
struct LazyNode{Name, D, C} <: AbstractMillNode
    data::D
    metadata::C

    function LazyNode{Name}(d::T, m::C=nothing) where {Name, T, C}
        new{Name, T, C}(d, m)
    end
end

"""
    LazyNode([Name::Symbol], d, m=nothing)
    LazyNode{Name}(d, m=nothing)

Construct a new [`LazyNode`](@ref) with name `Name`, data `d`, and metadata `m`.

# Examples
```jldoctest
julia> LazyNode(:Codons, ["GGGCGGCGA", "CCTCGCGGG"])
LazyNode{:Codons, Vector{String}, Nothing}:
 "GGGCGGCGA"
 "CCTCGCGGG"
```

See also: [`AbstractMillNode`](@ref), [`LazyModel`](@ref), [`Mill.unpack2mill`](@ref).
"""
LazyNode(Name::Symbol, d, m=nothing) = LazyNode{Name}(d, m)

"""
    Mill.unpack2mill(x::LazyNode)

Return a representation of [`LazyNode`](@ref) `x` using [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) structures. Every custom
[`LazyNode`](@ref) should have a special method as it is used in [`LazyModel`](@ref).

# Examples
```julia-repl
julia> function Mill.unpack2mill(ds::LazyNode{:Sentence})
    s = split.(ds.data, " ")
    x = NGramMatrix(reduce(vcat, s))
    BagNode(x, Mill.length2bags(length.(s)))
end;
```

```jldoctest
julia> LazyNode{:Sentence}(["foo bar", "baz"]) |> Mill.unpack2mill
BagNode  2 obs, 80 bytes
  ╰── ArrayNode(2053×3 NGramMatrix with Int64 elements)  3 obs, 170 bytes
```

See also: [`LazyNode`](@ref), [`LazyModel`](@ref).
"""
unpack2mill(::LazyNode{N}) where N = error("No `unpack2mill` method for LazyNode{$N}")

mapdata(f, x::LazyNode{N}) where N = LazyNode{N}(mapdata(f, x.data), x.metadata)

dropmeta(x::LazyNode{N}) where N = LazyNode{N}(x.data)

function Base.reduce(::typeof(catobs), as::Vector{<: LazyNode{N}}) where N
    LazyNode{N}(reduce(catobs, data.(as)), reduce(catobs, metadata.(as)))
end

Base.getindex(x::LazyNode{N, T}, i::VecOrRange{<:Int}) where {N, T} = LazyNode{N}(subset(x.data, i))

Base.hash(n::LazyNode, h::UInt) = hash((n.data), h)
(n1::LazyNode == n2::LazyNode) = n1.data == n2.data
Base.isequal(n1::LazyNode, n2::LazyNode) = isequal(n1.data, n2.data)
