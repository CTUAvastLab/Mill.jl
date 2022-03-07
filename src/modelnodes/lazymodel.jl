"""
    LazyModel{Name, T} <: AbstractMillModel

A model node for processing [`LazyNode`](@ref)s. It applies a (sub)model `m` stored in it to data of
the [`LazyNode`](@ref) after calling [`Mill.unpack2mill`](@ref).

# Examples
```jldoctest unpack2mill; output=false
function Mill.unpack2mill(ds::LazyNode{:Sentence})
    s = split.(ds.data, " ")
    x = NGramMatrix(reduce(vcat, s))
    BagNode(x, Mill.length2bags(length.(s)))
end
# output
```

```jldoctest unpack2mill
julia> Random.seed!(0);

julia> n = LazyNode{:Sentence}(["foo", "bar", "baz"])
LazyNode{:Sentence, Vector{String}, Nothing}:
 "foo"
 "bar"
 "baz"

julia> m = LazyModel{:Sentence}(BagModel(Dense(2053, 3), SegmentedMean(3), identity))
LazyModel{Sentence}
  └── BagModel ↦ SegmentedMean(3) ↦ identity 	# 1 arrays, 3 params (all zero), 52 bytes
        └── ArrayModel(Dense(2053, 3)) 	# 2 arrays, 6_162 params, 24.148 KiB
```

```jldoctest unpack2mill; filter=$(DOCTEST_FILTER)
julia> m(n)
3×3 Matrix{Float32}:
 -0.06... -0.03... -0.04...
  0.02...  0.00... -0.07...
 -0.00...  0.06... -0.07...
```

See also: [`AbstractMillModel`](@ref), [`LazyNode`](@ref), [`Mill.unpack2mill`](@ref).
"""
struct LazyModel{Name, T <: AbstractMillModel} <: AbstractMillModel
    m::T

    function LazyModel{Name}(m) where Name
        m = _arraymodel(m)
        new{Name, typeof(m)}(m)
    end
end

"""
    LazyModel([Name::Symbol], m::AbstractMillModel)
    LazyModel{Name}(m::AbstractMillModel)

Construct a new [`LazyModel`](@ref) with name `Name`, and model `m`.

It is also possible to pass any function as `m` instead of a model node. In that case,
it is wrapped into an [`ArrayNode`](@ref).

# Examples
```jldoctest
julia> LazyModel{:Sentence}(ArrayModel(Dense(2, 2)))
LazyModel{Sentence}
  └── ArrayModel(Dense(2, 2)) 	# 2 arrays, 6 params, 104 bytes

julia> LazyModel(:Sentence, Dense(2, 2))
LazyModel{Sentence}
  └── ArrayModel(Dense(2, 2)) 	# 2 arrays, 6 params, 104 bytes
```

See also: [`AbstractMillModel`](@ref), [`LazyNode`](@ref), [`Mill.unpack2mill`](@ref).
"""
LazyModel(Name::Symbol, m) = LazyModel{Name}(m)

Flux.@functor LazyModel

(m::LazyModel{Name})(x::LazyNode{Name}) where {Name} = m.m(unpack2mill(x))
