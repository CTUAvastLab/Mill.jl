"""
    ProductModel{T <: Mill.VecOrTupOrNTup{<:AbstractMillModel}, U} <: AbstractMillModel

A model node for processing [`ProductNode`](@ref)s. For each subtree of the data node it applies one
(sub)model from `ms` and then applies `m` on the concatenation of results.

# Examples
```jldoctest product_model
julia> Random.seed!(0);

julia> n = ProductNode(a=ArrayNode([0 1; 2 3]), b=ArrayNode([4 5; 6 7]))
ProductNode  # 2 obs, 16 bytes
  ├── a: ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes
  ╰── b: ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes

julia> m1 = ProductModel(a=ArrayModel(Dense(2, 2)), b=ArrayModel(Dense(2, 2)))
ProductModel ↦ identity
  ├── a: ArrayModel(Dense(2 => 2))  # 2 arrays, 6 params, 104 bytes
  ╰── b: ArrayModel(Dense(2 => 2))  # 2 arrays, 6 params, 104 bytes
```

```jldoctest product_model; filter=$DOCTEST_FILTER
julia> m1(n)
4×2 Matrix{Float32}:
 -2.36...  -3.58...
 -2.11...  -3.40...
 -6.31...  -7.61...
 -2.54...  -2.66...
```

```jldoctest product_model
julia> m2 = ProductModel(a=identity, b=identity)
ProductModel ↦ identity
  ├── a: ArrayModel(identity)
  ╰── b: ArrayModel(identity)

julia> m2(n)
4×2 Matrix{Int64}:
 0  1
 2  3
 4  5
 6  7
```

See also: [`AbstractMillModel`](@ref), [`AbstractProductNode`](@ref), [`ProductNode`](@ref).
"""
struct ProductModel{T<:VecOrTupOrNTup{AbstractMillModel},U} <: AbstractMillModel
    ms::T
    m::U

    function ProductModel(ms::Union{Tuple,NamedTuple,AbstractVector}, m=identity)
        ms = map(_arraymodel, ms)
        new{typeof(ms), typeof(m)}(ms, m)
    end
end

Flux.@functor ProductModel

"""
    ProductModel(ms, m=identity)
    ProductModel(m=identity; ms...)

Construct a [`ProductModel`](@ref) from the arguments. `ms` should an iterable
(`Tuple`, `NamedTuple` or `Vector`) of one or more [`AbstractMillModel`](@ref)s.

It is also possible to pass any function as elements of `ms`. In that case, it is wrapped into
an [`ArrayNode`](@ref).

# Examples
```jldoctest
julia> ProductModel(a=ArrayModel(Dense(2, 2)), b=identity)
ProductModel ↦ identity
  ├── a: ArrayModel(Dense(2 => 2))  # 2 arrays, 6 params, 104 bytes
  ╰── b: ArrayModel(identity)

julia> ProductModel(Dense(4, 2); a=ArrayModel(Dense(2, 2)), b=Dense(2, 2))
ProductModel ↦ Dense(4 => 2)  # 2 arrays, 10 params, 120 bytes
  ├── a: ArrayModel(Dense(2 => 2))  # 2 arrays, 6 params, 104 bytes
  ╰── b: ArrayModel(Dense(2 => 2))  # 2 arrays, 6 params, 104 bytes

julia> ProductModel((identity, BagModel(ArrayModel(Dense(2, 2)), SegmentedMean(2), identity)))
ProductModel ↦ identity
  ├── ArrayModel(identity)
  ╰── BagModel ↦ SegmentedMean(2) ↦ identity  # 1 arrays, 2 params (all zero), 48 bytes
        ╰── ArrayModel(Dense(2 => 2))  # 2 arrays, 6 params, 104 bytes

julia> ProductModel(identity)
ProductModel ↦ identity
  ╰── ArrayModel(identity)
```

See also: [`AbstractMillModel`](@ref), [`AbstractProductNode`](@ref), [`ProductNode`](@ref).
"""
ProductModel(ms, args...) = ProductModel(tuple(ms), args...)
ProductModel(args...; ns...) = ProductModel(NamedTuple(ns), args...)

Base.getindex(m::ProductModel, i::Symbol) = m.ms[i]
Base.keys(m::ProductModel) = keys(m.ms)
Base.haskey(m::ProductModel{<:NamedTuple}, k::Symbol) = haskey(m.ms, k)

(m::ProductModel{<:AbstractVector})(x::ProductNode{<:AbstractVector}) = m.m(vcat(map((sm, sx) -> sm(sx), m.ms, x.data)...))

@generated function (m::ProductModel{<:NamedTuple{KM}})(x::ProductNode{<:NamedTuple{KD}}) where {KM,KD}
    @assert issubset(KM, KD)
    chs = map(KM) do k
        :(m.ms.$k(x.data.$k))
    end
    quote
        m.m(vcat($(chs...)))
    end
end

@generated function (m::ProductModel{T})(x::ProductNode{U}) where {T<:Tuple,U<:Tuple}
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
