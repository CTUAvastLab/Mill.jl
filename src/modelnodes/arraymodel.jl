"""
struct ArrayModel{T <: MillFunction} <: AbstractMillModel
m::T
end

use a Chain, Dense, or any other function on an ArrayNode
"""
struct ArrayModel{T} <: AbstractMillModel
    m::T
end

Flux.@functor ArrayModel

(m::ArrayModel)(x::ArrayNode) = mapdata(x -> m.m(x), x)

identity_model() = ArrayModel(identity)
const IdentityModel = ArrayModel{typeof(identity)}
