"""
    ProductModel{T <: Mill.VecOrTupOrNTup{<:AbstractMillModel}, U <: ArrayModel} <: AbstractMillModel

A model node for processing [`ProductNode`](@ref)s. For each subtree of the data node it applies one
(sub)model from `ms` and then applies `m` on the concatenation of results.

# Examples
```jldoctest product_model
julia> Random.seed!(0);

julia> n = ProductNode((a=ArrayNode([0 1; 2 3]), b=ArrayNode([4 5; 6 7])))
ProductNode 	# 2 obs, 16 bytes
  ├── a: ArrayNode(2×2 Array with Int64 elements) 	# 2 obs, 80 bytes
  └── b: ArrayNode(2×2 Array with Int64 elements) 	# 2 obs, 80 bytes

julia> m1 = ProductModel((a=ArrayModel(Dense(2, 2)), b=ArrayModel(Dense(2, 2))))
ProductModel ↦ ArrayModel(identity)
  ├── a: ArrayModel(Dense(2, 2)) 	# 2 arrays, 6 params, 104 bytes
  └── b: ArrayModel(Dense(2, 2)) 	# 2 arrays, 6 params, 104 bytes
```

```jldoctest product_model; filter=$(DOCTEST_FILTER)
julia> m1(n)
4×2 ArrayNode{Matrix{Float32}, Nothing}:
 -1.284...  -1.254...
  1.802...   3.711...
 -4.036...  -5.013...
  0.587...   0.372...
```

```jldoctest product_model
julia> m2 = ProductModel((a=identity, b=identity))
ProductModel ↦ ArrayModel(identity)
  ├── a: ArrayModel(identity)
  └── b: ArrayModel(identity)

julia> m2(n)
4×2 ArrayNode{Matrix{Int64}, Nothing}:
 0  1
 2  3
 4  5
 6  7
```

See also: [`AbstractMillModel`](@ref), [`AbstractProductNode`](@ref), [`ProductNode`](@ref).
"""
struct ProductModel{T <: VecOrTupOrNTup{AbstractMillModel}, U <: ArrayModel} <: AbstractMillModel
    ms::T
    m::U

    function ProductModel(ms::Union{Tuple, NamedTuple, AbstractVector}, m=identity_model())
        ms = map(_make_mill_model, ms)
        m = _make_mill_model(m)
        new{typeof(ms), typeof(m)}(ms, m)
    end
end

Flux.@functor ProductModel

"""
    ProductModel(ms, m=identity_model())

Construct a [`ProductModel`](@ref) from the arguments. `ms` should an iterable
(`Tuple`, `NamedTuple` or `Vector`) of one or more [`AbstractMillModel`](@ref)s, and `m` should be
an [`ArrayModel`](@ref).

It is also possible to pass any function (`Flux.Dense`, `Flux.Chain`, `identity`...) as elements of `ms`
or as `m`. In that case, they are wrapped into an [`ArrayNode`](@ref).

If `ms` is [`AbstractMillModel`](@ref), a one-element `Tuple` is constructed from it.

# Examples
```jldoctest
julia> ProductModel((a=ArrayModel(Dense(2, 2)), b=identity))
ProductModel ↦ ArrayModel(identity)
  ├── a: ArrayModel(Dense(2, 2)) 	# 2 arrays, 6 params, 104 bytes
  └── b: ArrayModel(identity)

julia> ProductModel((identity_model(), BagModel(ArrayModel(Dense(2, 2)), SegmentedMean(2), identity)))
ProductModel ↦ ArrayModel(identity)
  ├── ArrayModel(identity)
  └── BagModel ↦ SegmentedMean(2) ↦ ArrayModel(identity) 	# 1 arrays, 2 params (all zero), 48 bytes
        └── ArrayModel(Dense(2, 2)) 	# 2 arrays, 6 params, 104 bytes

julia> ProductModel(identity)
ProductModel ↦ ArrayModel(identity)
  └── ArrayModel(identity)
```

See also: [`AbstractMillModel`](@ref), [`AbstractProductNode`](@ref), [`ProductNode`](@ref).
"""
ProductModel(ms, m=identity_model()) = ProductModel((ms,), m)

Base.getindex(m::ProductModel, i::Symbol) = m.ms[i]
Base.keys(m::ProductModel) = keys(m.ms)
Base.haskey(m::ProductModel{<:NamedTuple}, k::Symbol) = haskey(m.ms, k)

(m::ProductModel{<:AbstractVector})(x::ProductNode{<:AbstractVector}) = m.m(vcat(map((sm, sx) -> sm(sx), m.ms, x.data)...))

@generated function (m::ProductModel{<:NamedTuple{KM}})(x::ProductNode{<:NamedTuple{KD}}) where {KM, KD}
    @assert issubset(KM, KD)
    chs = map(KM) do k
        :(m.ms.$k(x.data.$k))
    end
    quote
        m.m(vcat($(chs...)))
    end
end

@generated function (m::ProductModel{T})(x::ProductNode{U}) where {T <: Tuple, U <: Tuple}
    l1 = T.parameters |> length
    l2 = U.parameters |> length
    @assert l1 ≤ l2 "Applied ProductModel{<:Tuple} has more children than ProductNode"
    chs = map(1:l1) do i
        :(m.ms[$i](x.data[$i]))
    end
    quote
        m.m(vcat($(chs...)))
    end
end
