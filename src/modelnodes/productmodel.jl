"""
    ProductModel{T <: VecOrTupOrNTup{<:AbstractMillModel}, U <: ArrayModel} <: AbstractMillModel

A model node for processing [`ProductNode`](@ref)s. For each subtree of the data node it applies one
(sub)model from `ms` and then applies `m` on the concatenation of results.

# Examples
```jldoctest product_model
julia> Random.seed!(0);

julia> n = ProductNode((a=ArrayNode([0 1; 2 3]), b=ArrayNode([4 5; 6 7])))
ProductNode 	# 2 obs, 16 bytes
  ├── a: ArrayNode(2×2 Array with Int64 elements) 	# 2 obs, 80 bytes
  └── b: ArrayNode(2×2 Array with Int64 elements) 	# 2 obs, 80 bytes

julia> m1 = ProductModel((a=ArrayModel(TurboDense(2, 2)), b=ArrayModel(TurboDense(2, 2))))
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
julia> ProductModel((a=ArrayModel(TurboDense(2, 2)), b=identity))
ProductModel ↦ ArrayModel(identity)
  ├── a: ArrayModel(Dense(2, 2)) 	# 2 arrays, 6 params, 104 bytes
  └── b: ArrayModel(identity)

julia> ProductModel((identity_model(), BagModel(ArrayModel(TurboDense(2, 2)), SegmentedMean(2), identity)))
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
function ProductModel(ms::VecOrTupOrNTup{Union{MillFunction, AbstractMillModel}},
                                m::Union{MillFunction, ArrayModel}=identity_model())
    ProductModel(map(_make_array_model, ms), _make_array_model(m))
end
ProductModel(ms::Union{MillFunction, AbstractMillModel},
             m::Union{MillFunction, ArrayModel}=identity_model()) = ProductModel((ms,), m)

Base.getindex(m::ProductModel, i::Symbol) = m.ms[i]
Base.keys(m::ProductModel) = keys(m.ms)

using ManualMemory
function (f::ManualMemory.Reference{<:AbstractMillModel})(x)
    ManualMemory.dereference(f)(x)
end

function (m::ProductModel{<:Tuple})(x::ProductNode{<:Tuple})
    as = prodparts(m.ms, x)
    m.m(vcat(as...))
end

function prodparts(ms, x::ProductNode{<:Tuple})
    as = Array{Any, 1}(undef, length(ms))
    @batch for i in eachindex(ms)
        as[i] = ms[i](x.data[i])
    end
    as
end

Zygote.@adjoint function prodparts(ms::T, x::ProductNode{<:Tuple}) where T
    Δprodparts(__context__, ms, x)
end

function Δprodparts(cx::C, ms::T, x::ProductNode{<:Tuple}) where {C, T}
    plen = length(ms)
    ys_backs = Array{Any, 1}(undef, plen)
    cxs = fill(cx, plen)
    @batch for i in 1:plen
        ys_backs[i] = Zygote._pullback(cxs[i], ms[i], x.data[i])
    end
    if any(map(cx -> !isnothing(cx.cache), cxs))
        "derivatives w.r.t. global variables under parallel ProductModel are not supported" |> throw
    end
    if isempty(ys_backs)
        ys_backs, _ -> nothing
    else
        ys, backs = Zygote.unzip(ys_backs)
        ys, function (Δ)
            ds = Array{Any, 1}(undef, plen)
            @batch for i in 1:plen
                ds[i] = backs[i](Δ[i])
            end
            Δms, Δx = Zygote.unzip(ds)
            Δms = Tuple(Δms)
            Δx = Tuple(Δx)
            (Δms, (data=Δx, metadata=nothing))
        end
    end
end

###

function (m::ProductModel{<:NamedTuple})(x::ProductNode{<:NamedTuple})
    as = prodparts(m.ms, x)
    m.m(vcat(as...))
end

function prodparts(ms::T, x::ProductNode{<:NamedTuple}) where T
    as = Array{Any, 1}(undef, length(ms))
    @batch for i in 1:length(ms)
        as[i] = ms[i](x.data[i])
    end
    as
end

Zygote.@adjoint function prodparts(ms::T, x::ProductNode{<:NamedTuple}) where T
    Δprodparts(__context__, ms, x)
end

function Δprodparts(cx::C, ms::T, x::ProductNode{<:NamedTuple}) where {C, T}
    plen = length(ms)
    ys_backs = Array{Any, 1}(undef, plen)
    cxs = fill(cx, plen)
    @batch for i in 1:plen
        ys_backs[i] = Zygote._pullback(cxs[i], ms[i], x.data[i])
    end
    if any(map(cx -> !isnothing(cx.cache), cxs))
        "derivatives w.r.t. global variables under parallel ProductModel are not supported" |> throw
    end
    if isempty(ys_backs)
        ys_backs, _ -> nothing
    else
        ys, backs = Zygote.unzip(ys_backs)
        ys, function (Δ)
            ds = Array{Any, 1}(undef, plen)
            @batch for i in plen:-1:1
                ds[i] = backs[i](Δ[i])
            end
            Δms, Δx = Zygote.unzip(ds)
            Δms = Tuple(Δms)
            Δx = NamedTuple{keys(x)}(Δx)
            (Δms, (data=Δx, metadata=nothing))
        end
    end
end
