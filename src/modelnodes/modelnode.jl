using HierarchicalUtils: encode, stringify

abstract type AbstractMillModel end

const MillFunction = Union{Flux.Dense, Flux.Chain, Function}

include("arraymodel.jl")
include("bagmodel.jl")
include("productmodel.jl")
include("missingmodel.jl")
include("lazymodel.jl")

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
        # use for julia 1.5 and higher
        # im, d = only(ms)
        im, d = ms[1]
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
        # use for julia 1.5 and higher
        # im, d = only(ms)
        # ProductModel((; only(ks)=>im)), d
        im, d = ms[1]
        ProductModel((; ks[1]=>im)), d
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
    m = ArrayModel(t)
    m, size(m(x).data, 1)
end

function _reflectinmodel(ds::LazyNode{Name}, db, da, b, a, s, ski, sci) where Name
    pm, d = Mill._reflectinmodel(unpack2mill(ds), db, da, b, a, s * Mill.encode(1, 1), ski, sci)
    LazyModel{Name}(pm), d
end

function _reflectinmodel(x::MissingNode, db, da, b, a, s, ski, sci)
    im, d = _reflectinmodel(x.data, db, da, b, a, s, ski, sci)
    θ = zeros(Float32, d)
    MissingModel(im, θ), d
end
