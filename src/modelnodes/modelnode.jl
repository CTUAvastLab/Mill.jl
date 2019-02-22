abstract type MillModel end

const MillFunction = Union{Flux.Dense, Flux.Chain, Function}

Base.show(io::IO, m::MillModel) = modelprint(io, m, tr=false)
modelprint(io::IO, m::MillModel; tr=false, pad=[]) = paddedprint(io, m, "\n")

include("arraymodel.jl")
include("bagmodel.jl")
include("productmodel.jl")

reflectinmodel(x) = reflectinmodel(x, d->Flux.Dense(d, 10), Dict(), d->SegmentedMean(), Dict())
reflectinmodel(x, def_builder) = reflectinmodel(x, def_builder, Dict(), d->SegmentedMean(), Dict())
reflectinmodel(x, def_builder, builder_dict) = reflectinmodel(x, def_builder, builder_dict, d->SegmentedMean(), Dict())
reflectinmodel(x, def_builder, builder_dict, def_agg) = reflectinmodel(x, def_builder, builder_dict, def_agg, Dict())

function reflectinmodel(x::AbstractBagNode, b, bd, a, ad, s="")
    im, d = reflectinmodel(x.data, b, bd, a, ad, encode(s, 1, 1))
    c = stringify(s)
    agg = c in keys(ad) ? ad[c](d) : a(d)
    bm, d = reflectinmodel(BagModel(im, agg)(x), b, bd, a, ad, s)
    BagModel(im, agg, bm), d
end

function reflectinmodel(x::AbstractTreeNode, b, bd, a, ad, s="")
    n = length(x.data)
    mm = [reflectinmodel(xx, b, bd, a, ad, encode(s, i, n)) for (i, xx) in enumerate(x.data)]
    im = tuple([i[1] for i in mm]...)
    tm, d = reflectinmodel(ProductModel(im)(x), b, bd, a, ad, s)
    ProductModel(im, tm), d
end

function reflectinmodel(x::ArrayNode, b, bd, a, ad, s="")
    c = stringify(s)
    t = c in keys(bd) ? bd[c](size(x.data, 1)) : b(size(x.data, 1))
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
