"""
struct ArrayModel{T <: MillFunction} <: MillModel
m::T
end

use a Chain, Dense, or any other function on an ArrayNode
"""
struct ArrayModel{T} <: MillModel
    m::T
end

Flux.@functor ArrayModel

(m::ArrayModel)(x::ArrayNode) = mapdata(x -> m.m(x), x)

modelprint(io::IO, m::ArrayModel; pad=[], s="", tr=false) = paddedprint(io, "ArrayModel(", m.m, ")$(tr_repr(s, tr))")


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

Flux.activations(::typeof(identity), x::Array{Float32,2}) = (x,)