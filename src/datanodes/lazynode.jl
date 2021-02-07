"""
    LazyNode{Name, D, C} <: AbstractNode

Data node storing data of type `D` in a lazy manner and optional metadata of type `C`.

Source of data or its type is specified in `Name`.

See also: [`AbstractNode`](@ref), [`LazyModel`](@ref), [`Mill.unpack2mill`](@ref).
"""
struct LazyNode{Name, D, C} <: AbstractNode
    data::D
    metadata::C
end

"""
    LazyNode([Name::Symbol], d, m=nothing)
    LazyNode{Name}(d, m=nothing)

Construct a new [`LazyNode`](@ref) with name `Name`, data `d`, and metadata `m`.

# Examples
```jlddoctest
julia> LazyNode(:Codons, ["GGGCGGCGA", "CCTCGCGGG"])
LazyNode{Codons} with 2 obs
```

See also: [`AbstractNode`](@ref), [`LazyModel`](@ref), [`Mill.unpack2mill`](@ref).
"""
LazyNode(Name::Symbol, d::T, m::C=nothing) where {T, C} = LazyNode{Name, T, C}(d, m)
LazyNode{Name}(d::T, m::C=nothing) where {Name, T, C} = LazyNode{Name, T, C}(d, m)

Base.ndims(x::LazyNode) = Colon()
StatsBase.nobs(a::LazyNode) = length(a.data)
StatsBase.nobs(a::LazyNode, ::Type{ObsDim.Last}) = nobs(a.data)

function Base.reduce(::typeof(catobs), as::Vector{<: LazyNode{N}}) where N
    LazyNode{N}(reduce(catobs, data.(as)), reduce(catobs, metadata.(as)))
end

Base.getindex(x::LazyNode{N, T}, i::VecOrRange{<:Int}) where {N, T} = LazyNode{N}(subset(x.data, i))

Base.hash(e::LazyNode, h::UInt) = hash((e.data), h)
(e1::LazyNode == e2::LazyNode) = e1.data == e2.data
Base.isequal(e1::LazyNode, e2::LazyNode) = isequal(e1.data, e2.data)
