"""
struct BagModel{T <: MillModel, U <: MillModel} <: MillModel
im::T
a::Aggregation
bm::U
end

use a `im` model on data in `BagNode`, the uses function `a` to aggregate individual bags,
and finally it uses `bm` model on the output
"""
struct BagModel{T <: MillModel, A, U <: MillModel} <: MillModel
    im::T
    a::A
    bm::U
end

Flux.@treelike BagModel
# Flux.@functor BagModel

BagModel(im::MillFunction, a, bm::MillFunction) = BagModel(ArrayModel(im), a, ArrayModel(bm))
BagModel(im::MillModel, a, bm::MillFunction) = BagModel(im, a, ArrayModel(bm))
BagModel(im::MillFunction, a, bm::MillModel) = BagModel(ArrayModel(im), a, bm)
BagModel(im::MillFunction, a) = BagModel(im, a, identity)
BagModel(im::MillModel, a) = BagModel(im, a, ArrayModel(identity))

(m::BagModel)(x::WeightedBagNode{<: AbstractNode}) = m.bm(m.a(m.im(x.data), x.bags, x.weights))
# if the data is missing, we do not use the mapping
(m::BagModel)(x::WeightedBagNode{<: Missing}) = m.bm(ArrayNode(m.a(x.data, x.bags, x.weights)))

(m::BagModel)(x::BagNode{<: AbstractNode}) = m.bm(m.a(m.im(x.data), x.bags))
# if the data is missing, we do not use the mapping
(m::BagModel)(x::BagNode{<: Missing}) = m.bm(ArrayNode(m.a(x.data, x.bags)))

function modelprint(io::IO, m::BagModel{T, A}; pad=[], s="", tr=false) where {T, A}
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "BagModel$(tr_repr(s, tr))\n", color=c)
    paddedprint(io, "  ├── ", color=c, pad=pad)
    modelprint(io, m.im, pad=[pad; (c, "  │   ")], s=s * encode(1, 1), tr=tr)
    println(io);
    paddedprint(io, "  ├── ", color=c, pad=pad)
    modelprint(io, m.a, pad=[pad; (c, "  │   ")])
    # A == Aggregation || paddedprint(io, '\n', color=c)
    println(io);
    paddedprint(io, "  └── ", color=c, pad=pad)
    modelprint(io, m.bm, pad=[pad; (c, "  │   ")])
end
