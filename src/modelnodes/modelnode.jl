using HierarchicalUtils: encode, stringify

abstract type AbstractMillModel end

const MillFunction = Union{Flux.Dense, Flux.Chain, Function}

include("arraymodel.jl")
include("bagmodel.jl")
include("productmodel.jl")
include("missingmodel.jl")
include("lazymodel.jl")

reflectinmodel(x, db=d->Flux.Dense(d, 10), da=d->SegmentedMean(d); b = Dict(), a = Dict(),
               single_key_identity=true) = _reflectinmodel(x, db, da, b, a, "", single_key_identity)[1]

function _reflectinmodel(x::AbstractBagNode, db, da, b, a, s)
    im, d = _reflectinmodel(x.data, db, da, b, a, s * encode(1, 1))
    c = stringify(s)
    agg = c in keys(a) ? a[c](d) : da(d)
    bm, d = _reflectinmodel(BagModel(im, agg)(x), db, da, b, a, s)
    BagModel(im, agg, bm), d
end

function _reflectinmodel(x::AbstractProductNode, db, da, b, a, s)
    n = length(x.data)
    ms = [_reflectinmodel(xx, db, da, b, a, s * encode(i, n), ski) for (i, xx) in enumerate(x.data)]
    if ski && n == 1
        # use for julia 1.5 and higher
        # im, d = only(ms)
        im, d = ms[1]
        ProductModel(im), d
    else
        im = tuple([i[1] for i in ms]...)
        tm, d = _reflectinmodel(ProductModel(im)(x), db, da, b, a, s, ski)
        ProductModel(im, tm), d
    end
end

function _reflectinmodel(x::ProductNode{T,C}, db, da, b, a, s) where {T<:NamedTuple, C}
    n = length(x.data)
    ks = keys(x.data)
    ms = [_reflectinmodel(x.data[k], db, da, b, a, s * encode(i, n), ski) for (i, k) in enumerate(ks)]
    if ski && n == 1
        # use for julia 1.5 and higher
        # im, d = only(ms)
        # ProductModel((; only(ks)=>im)), d
        im, d = ms[1]
        ProductModel((; ks[1]=>im)), d
    else
        im = (; (k=>v[1] for (k,v) in zip(ks, ms))...)
        tm, d = _reflectinmodel(ProductModel(im)(x), db, da, b, a, s, ski)
        ProductModel(im, tm), d
    end
end

function _reflectinmodel(x::ArrayNode, db, da, b, a, s)
    c = stringify(s)
    t = c in keys(b) ? b[c](size(x.data, 1)) : db(size(x.data, 1))
    m = ArrayModel(t)
    m, size(m(x).data, 1)
end
