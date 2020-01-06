"""
struct ArrayModel{T <: MillFunction} <: MillModel
m::T
end

use a Chain, Dense, or any other function on an ArrayNode
"""
struct ArrayModel{T <: MillFunction} <: MillModel
    m::T
end

Flux.@functor ArrayModel

(m::ArrayModel)(x::ArrayNode) = mapdata(x -> m.m(x), x)

modelprint(io::IO, m::ArrayModel; pad=[], s="", tr=false) = paddedprint(io, "ArrayModel(", m.m, ")$(tr_repr(s, tr))")

