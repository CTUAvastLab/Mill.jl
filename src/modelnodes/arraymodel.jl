"""
    ArrayModel{T <: MillFunction} <: AbstractMillModel

A model node for processing [`ArrayNode`](@ref)s. It applies a (sub)model `m` stored in it to data of 
the [`ArrayNode`](@ref).

# Examples
```jldoctest array_model
julia> Random.seed!(0);
```

```jldoctest array_model; filter=$(DOCTEST_FILTER)
julia> n = ArrayNode(randn(Float32, 2, 2))
2×2 ArrayNode{Matrix{Float32}, Nothing}:
 0.679...  -0.353...
 0.828...  -0.135...
```

```jldoctest array_model
julia> m = ArrayModel(Dense(2, 2))
ArrayModel(Dense(2, 2))
```

```jldoctest array_model; filter=$(DOCTEST_FILTER)
julia> m(n)
2×2 ArrayNode{Matrix{Float32}, Nothing}:
 0.661...  -0.188...
 0.101...   0.275...
```

See also: [`AbstractMillModel`](@ref), [`IdentityModel`](@ref), [`identity_model`](@ref), [`ArrayNode`](@ref).
"""
struct ArrayModel{T <: MillFunction} <: AbstractMillModel
    m::T
end

Flux.@functor ArrayModel

# (m::ArrayModel)(x::ArrayNode) = ArrayNode(m.m(getfield(x, :data)))

function (m::ArrayModel)(x::ArrayNode) 
    c = getfield(m, :m)
    a = getfield(x, :data)
    ArrayNode(c(a))
end

"""
    identity_model()

Returns an [`ArrayModel`](@ref) realising the `identity` transformation.

# Examples
```jldoctest
julia> identity_model()
ArrayModel(identity)
```

See also: [`ArrayModel`](@ref), [`IdentityModel`](@ref).
"""
identity_model() = ArrayModel(identity)

"""
    IdentityModel

Alias for `ArrayModel{typeof(identity)}`.

See also: [`ArrayModel`](@ref), [`identity_model`](@ref).
"""
const IdentityModel = ArrayModel{typeof(identity)}

# Base.hash(m::ArrayModel{T}, h::UInt) where {T} = hash((T, m.m), h)
# (m1::ArrayModel{T} == m2::ArrayModel{T}) where {T} = m1.m == m2.m
