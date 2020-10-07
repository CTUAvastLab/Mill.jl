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

function _reflectinmodel(x::AbstractBagNode, db, da, b, a, s, ski)
    im, d = _reflectinmodel(x.data, db, da, b, a, s * encode(1, 1), ski)
    c = stringify(s)
    agg = c in keys(a) ? a[c](d) : da(d)
    bm, d = _reflectinmodel(BagModel(im, agg)(x), db, da, b, a, s, ski)
    BagModel(im, agg, bm), d
end

function _reflectinmodel(x::AbstractProductNode, db, da, b, a, s, ski)
    n = length(x.data)
    mm = [_reflectinmodel(xx, db, da, b, a, s * encode(i, n), ski) for (i, xx) in enumerate(x.data)]
    if ski && n == 1
        im, d = mm[1]
        ProductModel(im), d
    else
        im = tuple([i[1] for i in mm]...)
        tm, d = _reflectinmodel(ProductModel(im)(x), db, da, b, a, s, ski)
        ProductModel(im, tm), d
    end
end

function _reflectinmodel(x::ProductNode{T}, db, da, b, a, s, ski) where T <: NamedTuple
    n = length(x.data)
    ks = keys(x.data)
    mm = [_reflectinmodel(x.data[k], db, da, b, a, s * encode(i, n), ski) for (i, k) in enumerate(ks)]
    if ski && n == 1
        im, d = mm[1]
        ProductModel((; ks[1]=>im)), d
    else
        im = (; (k=>v[1] for (k,v) in zip(ks, mm))...)
        tm, d = _reflectinmodel(ProductModel(im)(x), db, da, b, a, s, ski)
        ProductModel(im, tm), d
    end
end

function _reflectinmodel(x::ArrayNode, db, da, b, a, s, ski)
    c = stringify(s)
    t = c in keys(b) ? b[c](size(x.data, 1)) : db(size(x.data, 1))
    m = ArrayModel(t)
    m, size(m(x).data, 1)
end

function _reflectinmodel(ds::LazyNode{Name}, db, da, b, a, s, ski) where Name
    pm, d = Mill._reflectinmodel(unpack2mill(ds), db, da, b, a, s * Mill.encode(1, 1), ski)
    LazyModel{Name}(pm), d
end

function _reflectinmodel(x::MissingNode, db, da, b, a, s, ski)
    im, d = _reflectinmodel(x.data, db, da, b, a, s, ski)
    θ = zeros(Float32, d)
    MissingModel(im, θ), d
end
