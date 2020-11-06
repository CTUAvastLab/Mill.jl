using HierarchicalUtils: encode, stringify

abstract type AbstractMillModel end

const MillFunction = Union{Dense, Chain, Function}

include("arraymodel.jl")
include("bagmodel.jl")
include("productmodel.jl")
include("lazymodel.jl")

function Base.show(io::IO, @nospecialize m::T) where T <: AbstractMillModel
    print(io, nameof(T))
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

function _reflectinmodel(x::AbstractBagNode, db, da, b, a, s, ski, sci)
    im, d = _reflectinmodel(x.data, db, da, b, a, s * encode(1, 1), ski, sci)
    c = stringify(s)
    agg = c in keys(a) ? a[c](d) : da(d)
    bm, d = _reflectinmodel(BagModel(im, agg)(x), db, da, b, a, s, ski, sci)
    BagModel(im, agg, bm), d
end

function _reflectinmodel(x::AbstractProductNode, db, da, b, a, s, ski, sci)
    n = length(x.data)
    ms = [_reflectinmodel(xx, db, da, b, a, s * encode(i, n), ski, sci) for (i, xx) in enumerate(x.data)]
    if ski && n == 1
        im, d = only(ms)
        ProductModel(im), d
    else
        im = tuple([i[1] for i in ms]...)
        tm, d = _reflectinmodel(ProductModel(im)(x), db, da, b, a, s, ski, sci)
        ProductModel(im, tm), d
    end
end

function _reflectinmodel(x::ProductNode{T}, db, da, b, a, s, ski, sci) where T <: NamedTuple
    n = length(x.data)
    ks = keys(x.data)
    ms = [_reflectinmodel(x.data[k], db, da, b, a, s * encode(i, n), ski, sci) for (i, k) in enumerate(ks)]
    if ski && n == 1
        im, d = only(ms)
        ProductModel((; only(ks)=>im)), d
    else
        im = (; (k=>v[1] for (k,v) in zip(ks, ms))...)
        tm, d = _reflectinmodel(ProductModel(im)(x), db, da, b, a, s, ski, sci)
        ProductModel(im, tm), d
    end
end

function _reflectinmodel(x::ArrayNode, db, da, b, a, s, ski, sci)
    c = stringify(s)
    r = size(x.data, 1)
    if c in keys(b)
        t = b[c](r)
    elseif sci && r == 1
        t = identity
    else
        t = db(r)
    end
    m = ArrayModel(_make_imputing(x, t))
    m, size(m(x).data, 1)
end

_make_imputing(x::Chain, t) = Chain(x[1:end-1], _make_imputing(x[end], t))
_make_imputing(x, t) = t
function _make_imputing(x::ArrayNode{T}, t::Dense) where T <: Union{MaybeHotMatrix{Maybe{<:Integer}},
                                                                    MaybeHotVector{Missing},
                                                                    NGramMatrix{Maybe{<:Sequence}}}
    ColImputingDense(x)
end
function _make_imputing(x::ArrayNode{T}, t::Dense) where T <: AbstractArray{Maybe{<:Number}}
    RowImputingDense(x)
end

function _reflectinmodel(ds::LazyNode{Name}, db, da, b, a, s, ski, sci) where Name
    pm, d = Mill._reflectinmodel(unpack2mill(ds), db, da, b, a, s * Mill.encode(1, 1), ski, sci)
    LazyModel{Name}(pm), d
end
