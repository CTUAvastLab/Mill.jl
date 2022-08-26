"""
    ArrayModel{T} <: AbstractMillModel

A model node for processing [`ArrayNode`](@ref)s. It applies a (sub)model `m` stored in it to data in 
an [`ArrayNode`](@ref).

# Examples
```jldoctest array_model
julia> Random.seed!(0);
```

```jldoctest array_model; filter=$(DOCTEST_FILTER)
julia> n = ArrayNode(randn(Float32, 2, 2))
2×2 ArrayNode{Matrix{Float32}, Nothing}:
 0.94... 1.53...
 0.13... 0.12...
```

```jldoctest array_model
julia> m = ArrayModel(Dense(2, 2))
ArrayModel(Dense(2 => 2))  # 2 arrays, 6 params, 104 bytes
```

```jldoctest array_model; filter=$(DOCTEST_FILTER)
julia> m(n)
2×2 Matrix{Float32}:
 -0.50... -0.77...
  0.25...  0.49...
```

See also: [`AbstractMillModel`](@ref), [`ArrayNode`](@ref).
"""
struct ArrayModel{T} <: AbstractMillModel
    m::T
end

Flux.@functor ArrayModel

(m::ArrayModel)(x::ArrayNode) = m.m(x.data)

_arraymodel(m) = ArrayModel(m)
_arraymodel(m::AbstractMillModel) = m
