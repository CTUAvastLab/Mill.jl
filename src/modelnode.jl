abstract type MillModel end

"""
    struct ArrayModel{T <: MillFunction} <: MillModel
        m::T
    end

    use a Chain, Dense, or any other function on an ArrayNode
"""
struct ArrayModel{T <: MillFunction} <: MillModel
    m::T
end

"""
    struct BagModel{T <: MillModel, U <: MillModel} <: MillModel
        im::T
        a::Aggregation
        bm::U
    end

    use a `im` model on data in `BagNode`, the uses function `a` to aggregate individual bags,
    and finally it uses `bm` model on the output
"""
struct BagModel{T <: MillModel, U <: MillModel} <: MillModel
    im::T
    a::Aggregation
    bm::U
end

"""
    struct ProductModel{N, T <: MillFunction} <: MillModel
        ms::NTuple{N, MillModel}
        m::ArrayModel{T}
    end

    uses each model in `ms` on each data in `TreeNode`, concatenate the output and pass it to the chainmodel `m`
"""
struct ProductModel{N, T <: MillFunction} <: MillModel
    ms::NTuple{N, MillModel}
    m::ArrayModel{T}
end

Base.push!(m::ArrayModel{Flux.Chain}, l) = (push!(m.m.layers, l); m)
Base.push!(m::ProductModel{N, Flux.Chain}, l) where N = (push!(m.m, l); m)
Base.push!(m::BagModel{T, ArrayModel{Flux.Chain}}, l) where T = (push!(m.bm, l); m)

BagModel(im::MillFunction, a, bm::MillFunction) = BagModel(ArrayModel(im), a, ArrayModel(bm))
BagModel(im::MillModel, a, bm::MillFunction) = BagModel(im, a, ArrayModel(bm))
BagModel(im::MillFunction, a, bm::MillModel) = BagModel(ArrayModel(im), a, bm)
BagModel(im::MillFunction, a) = BagModel(im, a, identity)
BagModel(im::MillModel, a) = BagModel(im, a, ArrayModel(identity))
BagModel(im::MillModel, a, bm::MillModel) = BagModel(im, Aggregation(a), bm)

ProductModel(ms::NTuple{N, MillModel}) where N = ProductModel(ms, ArrayModel(identity))
ProductModel(ms, f::MillFunction) = ProductModel(ms, ArrayModel(f))

Flux.@treelike ArrayModel
Flux.@treelike BagModel
Flux.@treelike ProductModel

Adapt.adapt(T, m::ArrayModel) = ArrayModel(Adapt.adapt(T, m.m))
Adapt.adapt(T, m::BagModel) = BagModel(Adapt.adapt(T, m.im), Adapt.adapt(T, m.a), Adapt.adapt(T, m.bm))
Adapt.adapt(T, m::ProductModel) = ProductModel((map(m -> Adapt.adapt(T, m), m.ms)...,), Adapt.adapt(T, m.m))

(m::ArrayModel)(x::ArrayNode) = mapdata(x -> m.m(x), x)
# enforce the same length of ProductModel and TreeNode
(m::BagModel)(x::BagNode) = m.bm(m.a(m.im(x.data), x.bags))
(m::BagModel)(x::WeightedBagNode) = m.bm(m.a(m.im(x.data), x.bags, x.weights))
(m::ProductModel{N})(x::TreeNode{N}) where N = m.m(ArrayNode(vcat(map(f -> f[1](f[2]).data, zip(m.ms, x.data))...)))

################################################################################

function reflectinmodel(x::AbstractBagNode, layerbuilder, a=d ->segmented_mean)
    im, d = reflectinmodel(x.data, layerbuilder, a)
    bm, d = reflectinmodel(BagModel(im, a(d))(x), layerbuilder, a)
    BagModel(im, a(d), bm), d
end

function reflectinmodel(x::AbstractTreeNode, layerbuilder, a=d -> segmented_mean)
    mm = map(i -> reflectinmodel(i, layerbuilder, a), x.data)
    im = tuple(map(i -> i[1], mm)...)
    tm, d = reflectinmodel(ProductModel(im)(x), layerbuilder, a)
    ProductModel(im, tm), d
end

function reflectinmodel(x::ArrayNode, layerbuilder, a=d -> segmented_mean)
    m = ArrayModel(layerbuilder(size(x.data, 1)))
    m, size(m(x).data, 1)
end

################################################################################

Base.show(io::IO, m::MillModel) = modelprint(io, m)

modelprint(io::IO, m::MillModel; pad=[]) = paddedprint(io, m, "\n")

modelprint(io::IO, m::ArrayModel; pad=[]) = paddedprint(io, m.m, "\n")

function modelprint(io::IO, m::BagModel; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "BagModel\n", color=c)
    paddedprint(io, "  ├── ", color=c, pad=pad)
    modelprint(io, m.im, pad=[pad; (c, "  │   ")])
    paddedprint(io, "  ├── ", color=c, pad=pad)
    paddedprint(io, m.a, "\n")
    paddedprint(io, "  └── ", color=c, pad=pad)
    modelprint(io, m.bm, pad=[pad; (c, "  │   ")])
end

function modelprint(io::IO, m::ProductModel; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "ProductModel(\n", color=c)
    for i in 1:length(m.ms)-1
        paddedprint(io, "  ├── ", color=c, pad=pad)
        modelprint(io, m.ms[i], pad=[pad; (c, "  │   ")])
    end
    paddedprint(io, "  └── ", color=c, pad=pad)
    modelprint(io, m.ms[end], pad=[pad; (c, "      ")])

    paddedprint(io, ") ↦  ", color=c, pad=pad)
    modelprint(io, m.m, pad=[pad; (c, "")])
end
