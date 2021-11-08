"""
    LazyModel{Name, T} <: AbstractMillModel

A model node for processing [`LazyNode`](@ref)s. It applies a (sub)model `m` stored in it to data of
the [`LazyNode`](@ref) after calling [`Mill.unpack2mill`](@ref).

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
julia> Random.seed!(0);

julia> n = LazyNode{:Sentence}(["foo bar", "baz"])
LazyNode{Sentence} 	# 2 obs, 90 bytes

julia> m = LazyModel{:Sentence}(BagModel(Dense(2053, 3), SegmentedMean(3), identity))
LazyModel{Sentence}
  └── BagModel ↦ SegmentedMean(3) ↦ ArrayModel(identity) 	# 1 arrays, 3 params (all zero), 52 bytes
        └── ArrayModel(Dense(2053, 3)) 	# 2 arrays, 6_162 params, 24.148 KiB
```

```jldoctest unpack2mill; filter=$(DOCTEST_FILTER)
julia> m(n)
3×2 ArrayNode{Matrix{Float32}, Nothing}:
 -0.006...  -0.022...
  0.034...   0.055...
 -0.062...   0.071...
```

See also: [`AbstractMillModel`](@ref), [`LazyNode`](@ref), [`Mill.unpack2mill`](@ref).
"""
struct LazyModel{Name, T <: AbstractMillModel} <: AbstractMillModel
    m::T
end

"""
    LazyModel([Name::Symbol], m::AbstractMillModel)
    LazyModel{Name}(m::AbstractMillModel)

Construct a new [`LazyModel`](@ref) with name `Name`, and model `m`.

# Examples
```jldoctest
julia> LazyModel{:Sentence}(ArrayModel(Dense(2, 2)))
LazyModel{Sentence}
  └── ArrayModel(Dense(2, 2)) 	# 2 arrays, 6 params, 104 bytes
```

See also: [`AbstractMillModel`](@ref), [`LazyNode`](@ref), [`Mill.unpack2mill`](@ref).
"""
LazyModel(Name::Symbol, m::T) where T <: AbstractMillModel = LazyNode{Name, T}(m)
LazyModel{Name}(m::M) where {Name, M} = LazyModel{Name, M}(m)

Flux.@functor LazyModel

(m::LazyModel{Name})(x::LazyNode{Name}) where {Name} = m.m(unpack2mill(x))
(m::LazyModel{Name})(x::AbstractVector{<:LazyNode{Name}}) where {Name} = m(reduce(catobs, x))

function HiddenLayerModel(m::LazyModel{N}, ds::LazyNode{N}, n) where {N}
    hm, o = HiddenLayerModel(m.m, unpack2mill(ds), n)
    return(LazyModel{N}(hm), o )
end

function mapactivations(hm::LazyModel{N}, x::LazyNode{N}, m::LazyModel{N}) where {N}
    ho, o = mapactivations(hm.m, unpack2mill(x), m.m)
end

# Base.hash(m::LazyModel{T}, h::UInt) where {T} = hash((T, m.m), h)
# (m1::LazyModel{T} == m2::LazyModel{T}) where {T} = m1.m == m2.m
