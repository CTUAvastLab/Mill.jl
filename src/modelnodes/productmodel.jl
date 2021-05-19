"""
    ProductModel{T <: VecOrTupOrNTup{<:AbstractMillModel}, U <: ArrayModel} <: AbstractMillModel

A model node for processing [`ProductNode`](@ref)s. For each subtree of the data node it applies one
(sub)model from `ms` and then applies `m` on the concatenation of results.

# Examples
```jldoctest product_model
julia> Random.seed!(0);

julia> n = ProductNode((a=ArrayNode([0 1; 2 3]), b=ArrayNode([4 5; 6 7])))
ProductNode with 2 obs
  ├── a: ArrayNode(2×2 Array with Int64 elements) with 2 obs
  └── b: ArrayNode(2×2 Array with Int64 elements) with 2 obs

julia> m1 = ProductModel((a=ArrayModel(Dense(2, 2)), b=ArrayModel(Dense(2, 2))))
ProductModel … ↦ ArrayModel(identity)
  ├── a: ArrayModel(Dense(2, 2))
  └── b: ArrayModel(Dense(2, 2))
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
ProductModel … ↦ ArrayModel(identity)
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
ProductModel … ↦ ArrayModel(identity)
  ├── a: ArrayModel(Dense(2, 2))
  └── b: ArrayModel(identity)

julia> ProductModel((identity_model(), BagModel(ArrayModel(Dense(2, 2)), SegmentedMean(2), identity)))
ProductModel … ↦ ArrayModel(identity)
  ├── ArrayModel(identity)
  └── BagModel … ↦ SegmentedMean(2) ↦ ArrayModel(identity)
        └── ArrayModel(Dense(2, 2))

julia> ProductModel(identity)
ProductModel … ↦ ArrayModel(identity)
  └── ArrayModel(identity)
```

See also: [`AbstractMillModel`](@ref), [`AbstractProductNode`](@ref), [`ProductNode`](@ref).
"""
function ProductModel(ms::VecOrTupOrNTup{Union{MillFunction, AbstractMillModel}},
                                m::Union{MillFunction, ArrayModel}=identity_model())
    ProductModel(map(_make_array_model, ms), _make_array_model(m))
end
ProductModel(ms::Union{MillFunction, AbstractMillModel},
             m::Union{MillFunction, ArrayModel}=identity_model()) = ProductModel((ms,), m)

Base.getindex(m::ProductModel, i::Symbol) = m.ms[i]
Base.keys(m::ProductModel) = keys(m.ms)

function (m::ProductModel{<:Tuple})(x::ProductNode{<:Tuple})
    m.m(vcat(map((sm, sx) -> sm(sx), m.ms, x.data)...))
end
function (m::ProductModel{<:NamedTuple})(x::ProductNode{<:NamedTuple})
    m.m(vcat(map((sm, sx) -> sm(sx), m.ms, x.data)...))
end

function HiddenLayerModel(m::ProductModel, x::ProductNode, k::Int)
    ks = keys(m.ms)
    hxms = [HiddenLayerModel(m.ms[i], x.data[i], k) for i in keys(m.ms)]
    hms = (;[ks[i] => hxms[i][1] for i in 1:length(ks)]...)
    xms = reduce(vcat, [hxms[i][2] for i in 1:length(ks)])

    hm, o = HiddenLayerModel(m.m, xms, k)
    ProductModel(hms, hm), o
end

function mapactivations(hm::ProductModel, x::ProductNode, m::ProductModel)
    ks = keys(m.ms)
    _xxs = [mapactivations(hm.ms[i], x.data[i], m.ms[i]) for i in keys(m.ms)]
    hxs = foldl(+, [_xxs[i][1].data for i in 1:length(ks)])
    xxs = reduce(vcat, [_xxs[i][2] for i in 1:length(ks)])

    ho, o = mapactivations(hm.m, xxs, m.m)
    (ArrayNode(ho.data + hxs), o)
end

function fold(f, m::ProductModel, x)
    o₁ = map(k -> fold(f, m.ms[k], x.data[k]), keys(m.ms))
    o₂ = f(o₁)
    f(m.m, o₂)
end

# Base.hash(m::ProductModel{TT,T}, h::UInt) where {TT,T} = hash((TT, T, m.ms, m.m), h)
# (m1::ProductModel{TT,T} == m2::ProductModel{TT,T}) where {TT,T} = m1.ms == m2.ms && m1.m == m2.m
