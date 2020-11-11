using HierarchicalUtils: encode, stringify

abstract type AbstractMillModel end

const MillFunction = Union{Dense, Chain, Function}

include("arraymodel.jl")
include("bagmodel.jl")
include("productmodel.jl")
include("lazymodel.jl")

function Base.show(io::IO, @nospecialize(m::AbstractMillModel))
    print(io, nameof(typeof(m)))
    if !get(io, :compact, false)
        _show_submodels(io, m)
    end
end

_show_submodels(io, m::ArrayModel) = print(io, "(", m.m, ")")
_show_submodels(io, m::BagModel) = print(io, " ↦ ", m.a, " ↦ ", m.bm)
_show_submodels(io, m::ProductModel) = print(io, " ↦ ", m.m)
_show_submodels(io, m::LazyModel{Name}) where {Name} = print(io, "{", Name, "}")
_show_submodels(io, _) = print(io)

function reflectinmodel(x, db=d->Flux.Dense(d, 10), da=d->SegmentedMean(d); b = Dict(), a = Dict(),
               single_key_identity=true, single_scalar_identity=true)
    _reflectinmodel(x, db, da, b, a, "", single_key_identity, single_scalar_identity)[1]
end

function _reflectinmodel(x::AbstractBagNode, db, da, b, a, s, ski, ssi)
    im, d = _reflectinmodel(x.data, db, da, b, a, s * encode(1, 1), ski, ssi)
    c = stringify(s)
    agg = c in keys(a) ? a[c](d) : da(d)
    bm, d = _reflectinmodel(BagModel(im, agg)(x), db, da, b, a, s, ski, ssi)
    BagModel(im, agg, bm), d
end

function _reflectinmodel(x::AbstractProductNode, db, da, b, a, s, ski, ssi)
    n = length(x.data)
    ms = [_reflectinmodel(xx, db, da, b, a, s * encode(i, n), ski, ssi) for (i, xx) in enumerate(x.data)]
    if ski && n == 1
        im, d = only(ms)
        ProductModel(im), d
    else
        im = tuple([i[1] for i in ms]...)
        tm, d = _reflectinmodel(ProductModel(im)(x), db, da, b, a, s, ski, ssi)
        ProductModel(im, tm), d
    end
end

function _reflectinmodel(x::ProductNode{T}, db, da, b, a, s, ski, ssi) where T <: NamedTuple
    n = length(x.data)
    ks = keys(x.data)
    ms = [_reflectinmodel(x.data[k], db, da, b, a, s * encode(i, n), ski, ssi) for (i, k) in enumerate(ks)]
    if ski && n == 1
        im, d = only(ms)
        ProductModel((; only(ks)=>im)), d
    else
        im = (; (k=>v[1] for (k,v) in zip(ks, ms))...)
        tm, d = _reflectinmodel(ProductModel(im)(x), db, da, b, a, s, ski, ssi)
        ProductModel(im, tm), d
    end
end

function _reflectinmodel(x::ArrayNode, db, da, b, a, s, ski, ssi)
    c = stringify(s)
    r = size(x.data, 1)
    if c in keys(b)
        t = b[c](r) |> ArrayModel
    elseif ssi && r == 1
        t = identity_model()
    else
        t = db(r) |> ArrayModel
    end
    m = _make_imputing(x.data, t)
    m, size(m(x).data, 1)
end

_make_imputing(x, t) = t
_make_imputing(x, t::ArrayModel) = _make_imputing(x, t.m) |> ArrayModel
_make_imputing(x, t::Chain) = Chain(t[1:end-1], _make_imputing(x, t[end]))
_make_imputing(x::AbstractArray{Maybe{T}}, t::Dense) where T <: Number = PreImputingDense(t)
_make_imputing(x::MaybeHotVector{Missing}, t::Dense) = PostImputingDense(t)
_make_imputing(x::MaybeHotMatrix{Maybe{T}}, t::Dense) where T <: Integer = PostImputingDense(t)
_make_imputing(x::NGramMatrix{Maybe{T}}, t::Dense) where T <: Sequence = PostImputingDense(t)

_make_imputing(x, t::typeof(identity)) = t
function _make_imputing(x::AbstractArray{Maybe{T}}, ::typeof(identity)) where T <: Number
    PreImputingDense(IdentityDense(x))
end
function _make_imputing(x::MaybeHotVector{Missing}, ::typeof(identity))
    PostImputingDense(IdentityDense(x))
end
function _make_imputing(x::MaybeHotMatrix{Maybe{T}}, ::typeof(identity)) where T <: Integer
    PostImputingDense(IdentityDense(x))
end
function _make_imputing(x::NGramMatrix{Maybe{T}}, ::typeof(identity)) where T <: Sequence
    PostImputingDense(IdentityDense(x))
end

IdentityDense(x) = Dense(Matrix{Float32}(I, size(x, 1), size(x, 1)), zeros(Float32, size(x, 1)))

function _reflectinmodel(ds::LazyNode{Name}, db, da, b, a, s, ski, ssi) where Name
    pm, d = Mill._reflectinmodel(unpack2mill(ds), db, da, b, a, s * Mill.encode(1, 1), ski, ssi)
    LazyModel{Name}(pm), d
end
