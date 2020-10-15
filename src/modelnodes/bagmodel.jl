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