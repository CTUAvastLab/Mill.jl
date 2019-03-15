abstract type MillModel end

const MillFunction = Union{Flux.Dense, Flux.Chain, Function}

Base.show(io::IO, m::MillModel) = modelprint(io, m, tr=false)
modelprint(io::IO, m::MillModel; tr=false, pad=[]) = paddedprint(io, m, "\n")

include("arraymodel.jl")
include("bagmodel.jl")
include("productmodel.jl")

reflectinmodel(x, db, da = d->SegmentedMean(); b = Dict(), a = Dict()) = _reflectinmodel(x, db, da, b, a, "")

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

function _reflectinmodel(x::ArrayNode, db, da, b, a, s)
    c = stringify(s)
    t = c in keys(b) ? b[c](size(x.data, 1)) : db(size(x.data, 1))
    m = ArrayModel(t)
    m, size(m(x).data, 1)
end

# function reflectinmodel(x::AbstractBagNode, layerbuilder, a = d -> SegmentedMean())
#     im, d = reflectinmodel(x.data, layerbuilder, a)
#     bm, d = reflectinmodel(BagModel(im, a(d))(x), layerbuilder, a)
#     BagModel(im, a(d), bm), d
# end

# function reflectinmodel(x::AbstractTreeNode, layerbuilder, a = d -> SegmentedMean())
#     mm = [reflectinmodel(xx, layerbuilder, a) for xx in  x.data]
#     im = tuple([i[1] for i in mm]...)
#     tm, d = reflectinmodel(ProductModel(im)(x), layerbuilder, a)
#     ProductModel(im, tm), d
# end

# function reflectinmodel(x::ArrayNode, layerbuilder, a = d -> SegmentedMean())
#     m = ArrayModel(layerbuilder(size(x.data, 1)))
#     m, size(m(x).data, 1)
# end
