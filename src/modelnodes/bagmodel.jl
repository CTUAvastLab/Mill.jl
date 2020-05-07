"""
struct BagModel{T <: AbstractMillModel, U <: AbstractMillModel} <: AbstractMillModel
im::T
a::Aggregation
bm::U
end

use a `im` model on data in `BagNode`, the uses function `a` to aggregate individual bags,
and finally it uses `bm` model on the output
"""
struct BagModel{T <: AbstractMillModel, A, U <: AbstractMillModel} <: AbstractMillModel
    im::T
    a::A
    bm::U
end

Flux.@functor BagModel

BagModel(im::MillFunction, a, bm::MillFunction) = BagModel(ArrayModel(im), a, ArrayModel(bm))
BagModel(im::AbstractMillModel, a, bm::MillFunction) = BagModel(im, a, ArrayModel(bm))
BagModel(im::MillFunction, a, bm::AbstractMillModel) = BagModel(ArrayModel(im), a, bm)
BagModel(im::MillFunction, a) = BagModel(im, a, identity)
BagModel(im::AbstractMillModel, a) = BagModel(im, a, ArrayModel(identity))

(m::BagModel)(x::WeightedBagNode{<: AbstractNode}) = m.bm(m.a(m.im(x.data), x.bags, x.weights))

function (m::BagModel)(x::BagNode)
    ismissing(x.data) ? m.bm(ArrayNode(m.a(x.data, x.bags))) : m.bm(m.a(m.im(x.data), x.bags))
end

function HiddenLayerModel(m::BagModel, x::BagNode, k::Int)
    im, o = HiddenLayerModel(m.im, x.data, k)
    a = SegmentedMax(k)
    b = m.a(o, x.bags)
    bm, o = HiddenLayerModel(m.bm, b, k)
    BagModel(im, a, bm), o
end


function mapactivations(hm::BagModel, x::BagNode{M, B,C}, m::BagModel) where {M<: AbstractNode,B,C}
    hmi, mi = mapactivations(hm.im, x.data, m.im)
    ai = m.a(mi, x.bags)
    hai = hm.a(hmi, x.bags)
    hbo, bo = mapactivations(hm.bm, ai, m.bm)
    (ArrayNode(hbo.data + hai.data), bo)
end

function mapactivations(hm::BagModel, x::BagNode{M, B,C}, m::BagModel) where {M<: Missing,B,C}
    ai = m.a(missing, x.bags)
    hai = hm.a(missing, x.bags)
    hbo, bo = mapactivations(hm.bm, ArrayNode(ai), m.bm)
    (ArrayNode(hbo.data + hai), bo)
end

function fold(f, m::BagModel, x)
    o₁ = fold(f, m.im, x.data)
    o₂ = f(m.a, o₁, x.bags)
    o₃ = fold(f, m.bm, o₂)
    o₃
end