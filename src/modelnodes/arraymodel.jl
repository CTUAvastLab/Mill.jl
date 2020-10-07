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

function HiddenLayerModel(m::ArrayModel, x::ArrayNode, k::Int)
	os = Flux.activations(m.m, x.data)
	layers = map(x -> Dense(size(x,1), k), os)
	ArrayModel(layers), ArrayNode(os[end])
end


function mapactivations(hm::ArrayModel, x::ArrayNode, m::ArrayModel)
	os = Flux.activations(m.m, x.data)
	hx = mapfoldl((mx) -> mx[1](mx[2]),+,zip(hm.m, os))
	(ArrayNode(hx), ArrayNode(os[end]))
end

function fold(f, m::ArrayModel, x)
	f(m, x)
end

Flux.activations(::typeof(identity), x::Array{Float32,2}) = (x,)

# Base.hash(m::ArrayModel{T}, h::UInt) where {T} = hash((T, m.m), h)
# (m1::ArrayModel{T} == m2::ArrayModel{T}) where {T} = m1.m == m2.m
