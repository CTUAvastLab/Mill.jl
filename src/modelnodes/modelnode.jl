abstract type MillModel end

const MillFunction = Union{Flux.Dense, Flux.Chain, Function}

Base.show(io::IO, m::MillModel) = modelprint(io, m, tr=false)
modelprint(io::IO, m::MillModel; tr=false, pad=[]) = paddedprint(io, m, "\n")

include("arraymodel.jl")
include("bagmodel.jl")
include("productmodel.jl")

reflectinmodel(x, db, da=d->SegmentedMean(d); b = Dict(), a = Dict()) = _reflectinmodel(x, db, da, b, a, "")[1]

function _reflectinmodel(x::AbstractBagNode, db, da, b, a, s)
    im, d = _reflectinmodel(x.data, db, da, b, a, s * encode(1, 1))
    c = stringify(s)
    agg = c in keys(a) ? a[c](d) : da(d)
    bm, d = _reflectinmodel(BagModel(im, agg)(x), db, da, b, a, s)
    BagModel(im, agg, bm), d
end

function _reflectinmodel(x::AbstractTreeNode, db, da, b, a, s)
    n = length(x.data)
    mm = [_reflectinmodel(xx, db, da, b, a, s * encode(i, n)) for (i, xx) in enumerate(x.data)]
    im = tuple([i[1] for i in mm]...)
    tm, d = _reflectinmodel(ProductModel(im)(x), db, da, b, a, s)
    ProductModel(im, tm), d
end

function _reflectinmodel(x::TreeNode{T,C}, db, da, b, a, s) where {T<:NamedTuple, C}
    n = length(x.data)
    ks = keys(x.data)
    ms = (;[k => _reflectinmodel(x.data[k], db, da, b, a, s * encode(i, n))[1] for (i, k) in enumerate(ks)]...)
    @show ms
    tm, d = _reflectinmodel(ProductModel(ms)(x), db, da, b, a, s)
    ProductModel(ms, tm), d
end

function _reflectinmodel(x::ArrayNode, db, da, b, a, s)
    c = stringify(s)
    t = c in keys(b) ? b[c](size(x.data, 1)) : db(size(x.data, 1))
    m = ArrayModel(t)
    m, size(m(x).data, 1)
end
