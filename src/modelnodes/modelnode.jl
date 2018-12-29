abstract type MillModel end

const MillFunction = Union{Flux.Dense, Flux.Chain, Function}

Base.show(io::IO, m::MillModel) = modelprint(io, m)
modelprint(io::IO, m::MillModel; pad=[]) = paddedprint(io, m, "\n")

include("arraymodel.jl")
include("bagmodel.jl")
include("productmodel.jl")

function reflectinmodel(x::AbstractBagNode, layerbuilder, a = d -> SegmentedMean())
    im, d = reflectinmodel(x.data, layerbuilder, a)
    bm, d = reflectinmodel(BagModel(im, a(d))(x), layerbuilder, a)
    BagModel(im, a(d), bm), d
end

function reflectinmodel(x::AbstractTreeNode, layerbuilder, a = d -> SegmentedMean())
    mm = [reflectinmodel(xx, layerbuilder, a) for xx in  x.data]
    im = tuple([i[1] for i in mm]...)
    tm, d = reflectinmodel(ProductModel(im)(x), layerbuilder, a)
    ProductModel(im, tm), d
end

function reflectinmodel(x::ArrayNode, layerbuilder, a = d -> SegmentedMean())
    m = ArrayModel(layerbuilder(size(x.data, 1)))
    m, size(m(x).data, 1)
end
